from app.infra.config.providers import infra_config

class MinerUGateway:
    @property
    def base_url(self) -> str:
        """
         获取 MinerU 服务基础地址。

         Returns:
           str: MinerU 接口基础 URL。
        """
        return infra_config.mineru.base_url

    @property
    def api_key(self) -> str:
        """
         获取 MinerU 服务 API Token。

         Returns:
            str: MinerU 调用所需的 Token。
        """
        return infra_config.mineru.api_key

mineru_gateway = MinerUGateway()

# dataclass版本
from dataclasses import dataclass
from app.infra.config.providers import infra_config

@dataclass(frozen=True)  # frozen=True 代表只读，更安全！
class MinerUGateway:
    # 直接声明属性 + 默认值从配置读取
    base_url: str = infra_config.mineru.base_url
    api_key: str = infra_config.mineru.api_key

# 用法和你原来一模一样！
mineru_gateway = MinerUGateway()