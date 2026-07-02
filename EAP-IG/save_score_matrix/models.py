from pydantic import BaseModel, computed_field


class DatasetConfig(BaseModel):
    category: str = ''
    rephrase_type: str = 'only_stem'
    rephrase_model: str = 'gpt4o'
    num_samples: int = -1


class TargetModelConfig(BaseModel):
    model_name: str
    device: str = 'cuda'
    use_split_qkv_input: bool = True
    use_attn_result: bool = True
    use_hook_mlp_in: bool = True
    ungroup_grouped_query_attention: bool = True


class DiscoveryAlgConfig(BaseModel):
    topns: list[int]
    method: str = 'EAP-IG-inputs'
    steps: int = 20

    @computed_field
    @property
    def intervention(self) -> str:
        return 'zero' if self.method == 'EAP-IG-activations' else 'patching'
