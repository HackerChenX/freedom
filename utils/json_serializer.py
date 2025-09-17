"""
JSON序列化工具
解决Period枚举等对象的JSON序列化问题
"""

import json
from enum import Enum
from datetime import datetime, date
import pandas as pd
import numpy as np

class EnhancedJSONEncoder(json.JSONEncoder):
    """增强的JSON编码器，支持更多数据类型"""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, cls=EnhancedJSONEncoder, ensure_ascii=False, **kwargs)
    except Exception as e:
        # 如果序列化失败，转换为字符串
        return json.dumps(str(obj), ensure_ascii=False, **kwargs)

def safe_json_loads(json_str):
    """安全的JSON反序列化"""
    try:
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"JSON解析失败: {e}", "raw_data": json_str}
