#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from graphgpt.model.graph_layers import MPNN, GNN, CLIP, graph_transformer
from torch_geometric.data import Data
import json
import os.path as osp
import glob

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

class GraphLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"

class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path):
    """
    load pretrained CLIP model
    model_name:                 CLIP
    pretrain_model_path:        clip_gt_arxiv_pub(pretrained clip model)
    """
    # load conig json
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)                                       # load pretrainded state
    return model, args


def transfer_param_tograph(clip_graph, gnn):
    """
    transfer the graph model from pretrained clip to single gnn used for graphllama
    """
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn


class GraphLlamaModel(LlamaModel):                                  # inherit from LlamaModel
    config_class = GraphLlamaConfig
    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            """
            Transfer graph model from CLIP to GNN/Graph Transformer
            """
            if config.graph_tower == 'MPNN':                        # MPNN
                self.graph_tower = MPNN(in_channels = config.graph_hidden_size, hidden_channels = config.graph_hidden_size * 2, out_channels = config.graph_hidden_size, dropout = 0.1, num_layers = 2, if_param = False)
            elif config.graph_tower == "clip_gcn_arxiv":            # GNN
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = GNN(args)                                                    # new define single GNN
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)         # transfer clip model to single gnn
            elif config.graph_tower == "clip_gt":
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv": 
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv_pub": 
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)

        if hasattr(config, "use_graph_proj"):
            """
            Define projector
            """
            self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)

    def get_graph_tower(self):
        """
        Get graph model
        """
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                                  pretrain_graph_mlp_adapter=None, fsdp=None):
        """
        define graph_tower if it was not defined before, which is the graph model in graphllama
        """
        self.config.graph_tower = graph_tower
        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower == 'MPNN': 
                graph_tower = MPNN(in_channels = self.config.graph_hidden_size, hidden_channels = self.config.graph_hidden_size * 2, out_channels = self.config.graph_hidden_size, dropout = 0.1, num_layers = 2, if_param = False)
            elif self.config.graph_tower == "clip_gcn_arxiv": 
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = GNN(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt_arxiv":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt_arxiv_pub":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        else:
            graph_tower = self.graph_tower

        graph_tower.requires_grad_(False)               # freeze graph model

        if fsdp is not None and len(fsdp) > 0:          # ? multi graph model?
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        self.config.use_graph_proj = True                                           # need projector
        self.config.graph_select_layer = graph_select_layer
        self.config.graph_hidden_size = self.graph_tower.W_P.out_features           # dim of gnn output
        if not hasattr(self, 'graph_projector'):                                    # define projector
            self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:                                  # load pretrained projector
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:                                       # embeddings from llama model
            inputs_embeds = self.embed_tokens(input_ids)

        graph_tower = self.get_graph_tower()                            # get graph model
        if graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            with torch.no_grad():                                       # traverse graph data, get GNN output
                if type(graph_data) is list:
                    # variable length images
                    graph_node_features = []
                    if type(graph_data[0]) is Data:                     # pyg 中的data类型
                        for g in graph_data:
                            node_forward_out = graph_tower(g)
                            graph_node_features.append(node_forward_out)
                    elif type(graph_data[0]) is dict:
                        for g_dict in graph_data:
                            node_forward_out_1 = graph_tower(g_dict['graph_1'])
                            node_forward_out_2 = graph_tower(g_dict['graph_2'])
                            graph_node_features.append(node_forward_out_1)
                            graph_node_features.append(node_forward_out_2)
                else:
                    raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            
            # GNN ouput -> Projector
            if type(graph_data) is list:
                graph_node_features = [self.graph_projector(node_feature) for node_feature in graph_node_features]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            
            # ?? fake features of graph
            dummy_graph_features = torch.zeros(256, 128, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            new_input_embeds = []
            cur_graph_idx = 0
            # traverse every input sequences and its embeddings
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                # judge if it's graph-llm or single llm(no graph patch token)
                if (cur_input_ids == graph_tower.config.graph_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                # graph-llm
                if graph_tower.config.use_graph_start_end:                          # using graph start&end token
                    cur_graph_features = graph_node_features[cur_graph_idx]         # find node features
                    num_patches = cur_graph_features.shape[0]
                    # check if the num of start tokens and end tokens are equal
                    if (cur_input_ids == graph_tower.config.graph_start_token).sum() != (cur_input_ids == graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = torch.where(cur_input_ids == graph_tower.config.graph_start_token)[0]      # start token idx
                    # print(graph_start_tokens)
                    for graph_start_token_pos in graph_start_tokens:
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]                                                   # 
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), dim=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = torch.where(cur_input_ids == graph_tower.config.graph_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            # print(cur_graph_idx)
            # print(len(graph_node_features))
            assert cur_graph_idx == len(graph_node_features)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(GraphLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class GraphLlamaForCausalLM(LlamaForCausalLM):                                          # 继承生成式因果LLM model
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GraphLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_data = graph_data
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": [kwargs.get("graph_data", None)],
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs

    # 得到graph start\end\patch token id
    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                    tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)                                              # <g_patch>
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)        # <g_start>, <g_end>
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data                  # [32003, 4096]
                output_embeddings = self.get_output_embeddings().weight.data
                # init new token embeddings with avg of origional tokens embedding
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(             
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:                                                      # True
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():                          # not freeze input embeddings
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():                         # freeze ouput embeddings
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:                                                  # False
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]

AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)
