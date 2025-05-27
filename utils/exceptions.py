"""
异常类型模块

定义系统中使用的所有自定义异常类型
"""

class BaseError(Exception):
    """所有自定义异常的基类"""
    def __init__(self, message=None, details=None):
        self.message = message or "发生错误"
        self.details = details
        super().__init__(self.message)
        
    def __str__(self):
        if self.details:
            return f"{self.message}，详情: {self.details}"
        return self.message


# 策略相关异常
class StrategyError(BaseError):
    """策略相关错误的基类"""
    pass

class StrategyParseError(StrategyError):
    """策略解析错误"""
    pass

class StrategyExecutionError(StrategyError):
    """策略执行错误"""
    pass

class StrategyValidationError(StrategyError):
    """策略验证错误"""
    pass

class StrategyNotFoundError(StrategyError):
    """策略不存在错误"""
    pass


# 数据相关异常
class DataError(BaseError):
    """数据相关错误的基类"""
    pass

class DataAccessError(DataError):
    """数据访问错误"""
    pass

class DataValidationError(DataError):
    """数据验证错误"""
    pass

class DataNotFoundError(DataError):
    """数据不存在错误"""
    pass


# 指标相关异常
class IndicatorError(BaseError):
    """指标相关错误的基类"""
    pass

class IndicatorCalculationError(IndicatorError):
    """指标计算错误"""
    pass

class IndicatorExecutionError(IndicatorError):
    """指标执行错误"""
    pass

class IndicatorParameterError(IndicatorError):
    """指标参数错误"""
    pass

class IndicatorNotFoundError(IndicatorError):
    """指标不存在错误"""
    pass


# 配置相关异常
class ConfigError(BaseError):
    """配置相关错误的基类"""
    pass

class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass

class ConfigFileError(ConfigError):
    """配置文件错误"""
    pass


# 资源相关异常
class ResourceError(BaseError):
    """资源相关错误的基类"""
    pass

class ResourceNotFoundError(ResourceError):
    """资源不存在错误"""
    pass

class ResourceExhaustedError(ResourceError):
    """资源耗尽错误"""
    pass


# 权限相关异常
class PermissionError(BaseError):
    """权限相关错误的基类"""
    pass

class AuthenticationError(PermissionError):
    """认证错误"""
    pass

class AuthorizationError(PermissionError):
    """授权错误"""
    pass 