from google.adk.models.base_llm import BaseModel
import boto3

class AWSBedrockModel(BaseModel):
    """
    适配 AWS Bedrock 的 BaseModel
    可以直接传给 ADK Agent 的 model 参数
    """
    def __init__(self, model_id: str, region_name: str = "us-west-2"):
        super().__init__()
        self.model_id = model_id
        self.client = boto3.client("bedrock", region_name=region_name)

    def generate(self, prompt: str, **kwargs):
        """
        兼容 ADK BaseModel 的 generate 接口
        """
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=prompt.encode("utf-8"),
            contentType="text/plain"
        )
        return response['body'].read().decode("utf-8")
