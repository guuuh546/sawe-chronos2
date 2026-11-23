import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional


class TimeSeriesDataset(Dataset):
    """
    通用时序数据集
    支持窗口滑动窗口划分
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 96,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        """
        Args:
            data: [num_samples, num_features] 或 [num_samples]
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            normalize: 是否做归一化
            mean: 外部传入的均值（用于验证/测试集，使用训练集的统计量）
            std: 外部传入的标准差（用于验证/测试集，使用训练集的统计量）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 确保数据形状
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.data = data
        
        # 归一化
        if normalize:
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std
            else:
                self.mean = data.mean(axis=0)
                self.std = data.std(axis=0) + 1e-6
            self.data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros(data.shape[1])
            self.std = np.ones(data.shape[1])
        
        # 预转换为张量：避免 __getitem__ 中逐样本 numpy→tensor 拷贝（大数据集下的性能瓶颈）
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        # 计算窗口数量，确保非负
        self.num_windows = max(0, len(self.data) - seq_len - pred_len + 1)
        if self.num_windows == 0:
            raise ValueError(
                f"数据不足！需要至少 {seq_len + pred_len} 个样本，但只有 {len(self.data)} 个。"
                f"seq_len={seq_len}, pred_len={pred_len}"
            )
    
    def __len__(self) -> int:
        return self.num_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [seq_len, num_features]
            y: [pred_len, num_features]
        """
        x_start = idx
        x_end = idx + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len
        
        x = self.data[x_start:x_end]
        y = self.data[y_start:y_end]
        
        return x, y


def create_dataloaders(
    data: np.ndarray,
    seq_len: int = 96,
    pred_len: int = 96,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    normalize: bool = True,
    train_data_ratio: float = 1.0
):
    """
    创建训练/验证/测试集 DataLoader
    默认7:1:2划分
    支持少样本训练：train_data_ratio 控制使用多少比例的训练数据 (0.05, 0.1, 1.0 等)
    """
    total_len = len(data)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_data_full = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # 少样本训练：只取部分训练数据
    if train_data_ratio < 1.0:
        train_data = train_data_full[:int(len(train_data_full) * train_data_ratio)]
    else:
        train_data = train_data_full
    
    train_dataset_full = TimeSeriesDataset(train_data, seq_len, pred_len, normalize=normalize)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len, normalize=normalize,
                                     mean=train_dataset_full.mean, std=train_dataset_full.std)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len, normalize=normalize,
                                      mean=train_dataset_full.mean, std=train_dataset_full.std)
    
    train_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset_full,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def generate_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 1,
    seasonality_period: int = 24,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    生成合成时序数据（用于演示和测试）
    包含周期性、趋势和噪声
    """
    t = np.arange(num_samples)
    
    # 趋势项
    trend = 0.001 * t
    # 季节项
    seasonality = np.sin(2 * np.pi * t / seasonality_period)
    # 噪声
    noise = noise_level * np.random.randn(num_samples, num_features)
    
    data = trend.reshape(-1, 1) + seasonality.reshape(-1, 1) + noise
    return data


def load_statsmodels_dataset(name: str = "co2") -> np.ndarray:
    """
    加载 statsmodels 内置小数据集
    
    Args:
        name: 数据集名称，可选：
            - "co2": 大气CO2浓度 (经典时序数据，强季节性
            - "sunspots": 太阳黑子数据
            - "airline": 航空乘客数据（使用seaborn或statsmodels
            - "gold": 黄金价格数据
    
    Returns:
        时序数据 [num_samples, 1]
    """
    try:
        import statsmodels.datasets as sm_ds
        
        if name == "co2":
            data = sm_ds.co2.load_pandas().data
            # 处理可能的缺失值
            if hasattr(data, 'co2'):
                values = data['co2'].values
            else:
                values = data.iloc[:, 0].values
            # 填充缺失值
            import pandas as pd
            values = pd.Series(values).interpolate().values
        elif name == "sunspots":
            data = sm_ds.sunspots.load_pandas().data
            values = data['SUNACTIVITY'].values
        elif name == "airline":
            # 使用 seaborn 或手动加载经典航空乘客数据
            try:
                import seaborn as sns
                data = sns.load_dataset("flights")
                values = data['passengers'].values
            except:
                # 手动创建一个小型航空乘客数据
                values = np.array([112,118,132,129,121,135,148,148,136,119,104,118,
                                  115,126,141,135,125,149,170,170,158,133,114,140,
                                  145,150,178,163,172,178,199,199,184,162,146,166,
                                  171,180,193,181,183,218,230,242,209,191,172,194,
                                  196,196,236,235,229,243,264,272,237,211,180,201,
                                  204,188,235,227,234,264,302,293,259,229,203,229,
                                  242,233,267,269,270,315,364,347,312,274,237,278,
                                  284,277,317,313,318,374,413,405,355,306,271,306,
                                  315,301,356,348,355,422,465,467,404,347,305,336,
                                  340,318,362,348,363,435,491,505,404,359,310,337,
                                  360,342,406,396,420,472,548,559,463,407,362,405,
                                  417,391,419,461,472,535,622,606,508,461,390,432])
        elif name == "gold":
            # 使用 yfinance 或创建模拟黄金数据
            values = 1000 + np.cumsum(np.random.randn(500)*10)
        else:
            raise ValueError(f"未知数据集: {name}")
        
        # 确保数据量足够
        min_samples = 200
        if len(values) < min_samples:
            print(f"  警告：原始数据只有 {len(values)} 个样本，自动扩充到 {min_samples} 个")
            # 复制数据并添加少量噪声
            repeats = (min_samples // len(values)) + 1
            values = np.tile(values, repeats)[:min_samples]
            # 添加少量噪声
            values = values + np.random.randn(len(values)) * (values.std() * 0.05)
        
        return values.reshape(-1, 1)
    except Exception as e:
        print(f"加载数据集 {name} 失败: {e}")
        print("使用合成数据代替")
        return generate_synthetic_data()


def load_ett_dataset(
    dataset_name: str = "ETTh1",
    data_dir: str = "data/ETDataset"
) -> np.ndarray:
    """
    加载 ETT 数据集
    
    Args:
        dataset_name: 可选：
            - "ETTh1": 小时级数据 station1
            - "ETTh2": 小时级数据 station2
            - "ETTm1": 分钟级数据 station1
            - "ETTm2": 分钟级数据 station2
        data_dir: 数据集目录
    
    Returns:
        时序数据 [num_samples, num_features] (target 或全部特征
    """
    csv_path = os.path.join(data_dir, dataset_name + ".csv")
    
    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        print(f"请先从 https://github.com/zhouhaoyi/ETDataset 下载数据集")
        print("使用合成数据代替")
        return generate_synthetic_data()
    
    try:
        df = pd.read_csv(csv_path)
        
        # 特征列：HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        # OT 是目标值（油温）
        data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values
        
        print(f"加载 {dataset_name} 成功！")
        print(f"  样本数: {len(data)}")
        print(f"  特征数: {data.shape[1]}")
        print(f"  时间范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
        
        return data
    except Exception as e:
        print(f"加载 ETT 数据集失败: {e}")
        print("使用合成数据代替")
        return generate_synthetic_data()


def load_electricity_dataset(
    data_dir: str = "data/electricity-20260606T074431Z-3-001/electricity"
) -> np.ndarray:
    """
    加载 Electricity 数据集（UCI，370 户家庭用电，小时级）
    
    Returns:
        时序数据 [num_samples, num_features]
    """
    csv_path = os.path.join(data_dir, "electricity.csv")
    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        return generate_synthetic_data()

    try:
        df = pd.read_csv(csv_path)
        # 去掉 date 列，其余全是特征
        feature_cols = [c for c in df.columns if c != 'date']
        data = df[feature_cols].values.astype(np.float32)
        print(f"加载 Electricity 成功！样本数: {len(data)}, 特征数: {data.shape[1]}")
        return data
    except Exception as e:
        print(f"加载 Electricity 失败: {e}")
        return generate_synthetic_data()


def load_exchange_rate_dataset(
    data_dir: str = "data/exchange_rate-20260606T075627Z-3-001/exchange_rate"
) -> np.ndarray:
    """
    加载 Exchange Rate 数据集（8 国汇率，日级）
    
    Returns:
        时序数据 [num_samples, num_features]
    """
    csv_path = os.path.join(data_dir, "exchange_rate.csv")
    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        return generate_synthetic_data()

    try:
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != 'date']
        data = df[feature_cols].values.astype(np.float32)
        print(f"加载 Exchange Rate 成功！样本数: {len(data)}, 特征数: {data.shape[1]}")
        return data
    except Exception as e:
        print(f"加载 Exchange Rate 失败: {e}")
        return generate_synthetic_data()


def load_weather_dataset(
    data_path: str = "data/weather.csv"
) -> np.ndarray:
    """加载 Weather 数据集（21 个气象变量，10分钟级）"""
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return generate_synthetic_data()
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != 'date']
    data = df[feature_cols].values.astype(np.float32)
    print(f"加载 Weather 成功！样本数: {len(data)}, 特征数: {data.shape[1]}")
    return data


def load_ili_dataset(
    data_path: str = "data/national_illness.csv"
) -> np.ndarray:
    """加载 ILI 数据集（CDC 流感数据，周级，小样本）"""
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return generate_synthetic_data()
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != 'date']
    data = df[feature_cols].values.astype(np.float32)
    print(f"加载 ILI 成功！样本数: {len(data)}, 特征数: {data.shape[1]}")
    return data


def get_dataloaders_for_experiment(
    dataset_type: str = "synthetic",
    seq_len: int = 96,
    pred_len: int = 96,
    auto_adjust: bool = True,
    **kwargs
):
    """
    一键获取实验用的 DataLoaders
    
    Args:
        dataset_type: "synthetic" | "co2" | "sunspots" | "airline" | "gold" | "ETTh1" | "ETTm1"
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        auto_adjust: 是否自动调整 seq_len/pred_len 以适应数据长度
        **kwargs: 其他传递给 create_dataloaders 的参数
    
    Returns:
        dataloaders 字典
    """
    if dataset_type == "synthetic":
        data = generate_synthetic_data()
    elif dataset_type in ["co2", "sunspots", "airline", "gold"]:
        data = load_statsmodels_dataset(dataset_type)
    elif dataset_type in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        data = load_ett_dataset(dataset_type)
    elif dataset_type.lower() == "electricity":
        data = load_electricity_dataset()
    elif dataset_type.lower() == "exchange_rate":
        data = load_exchange_rate_dataset()
    elif dataset_type.lower() == "weather":
        data = load_weather_dataset()
    elif dataset_type.lower() == "illness":
        data = load_ili_dataset()
    else:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    
    # 自动调整以适应数据长度
    if auto_adjust:
        min_required = seq_len + pred_len + 1
        if len(data) < min_required:
            # 计算最大可用的 seq_len 和 pred_len
            total_available = len(data) - 1
            # 保持比例，pred_len 约为 seq_len 的一半
            new_pred_len = min(pred_len, total_available // 3)
            new_seq_len = min(seq_len, total_available - new_pred_len)
            
            if new_seq_len < 4 or new_pred_len < 2:
                # 数据太少，使用合成数据代替
                print(f"  警告：数据集 {dataset_type} 只有 {len(data)} 个样本，不足！")
                print(f"  使用合成数据代替")
                data = generate_synthetic_data(num_samples=1000)
            else:
                print(f"  警告：调整 seq_len={new_seq_len}, pred_len={new_pred_len} 以适应数据长度")
                seq_len, pred_len = new_seq_len, new_pred_len
    
    # 先检查一下数据是否足够，确保不会报错
    while len(data) < seq_len + pred_len + 1 and seq_len > 4 and pred_len > 2:
        seq_len = max(4, seq_len - 4)
        pred_len = max(2, pred_len - 2)
    
    if len(data) < seq_len + pred_len + 1:
        # 还是不够，用合成数据
        print(f"  警告：数据仍不足，使用合成数据代替")
        data = generate_synthetic_data(num_samples=1000)
    
    try:
        return create_dataloaders(data, seq_len=seq_len, pred_len=pred_len, **kwargs)
    except Exception as e:
        print(f"  创建 DataLoader 失败: {e}")
        print(f"  使用合成数据代替")
        data = generate_synthetic_data(num_samples=1000)
        return create_dataloaders(data, seq_len=24, pred_len=12, **kwargs)

