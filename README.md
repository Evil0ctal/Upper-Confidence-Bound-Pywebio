# UCB 算法在线演示

该仓库包含基于 PyWebIO 的 **UCB（上置信界）算法** 在线演示，UCB 算法常用于多臂老虎机问题，以优化决策并最大化累积奖励。演示包括自动 UCB 算法模拟和交互式手动策略对比。

## 功能

- 支持 **LaTeX 公式渲染**，帮助理解 UCB 算法。
- 实时模拟 UCB 算法，并以图表形式展示结果。
- 交互模式，手动选择拉动臂，并与 UCB 算法进行效果对比。

## 截图

![](https://raw.githubusercontent.com/Evil0ctal/Upper-Confidence-Bound-Pywebio/refs/heads/main/screenshots/2024-09-21_18-09-17.png)

![](https://raw.githubusercontent.com/Evil0ctal/Upper-Confidence-Bound-Pywebio/refs/heads/main/screenshots/09-21-2024_06_28.png)

## 快速开始

### 环境要求

在运行演示之前，请确保已安装 Python 3.x。

### 安装步骤

1. 克隆此仓库到本地：

   ```bash
   git clone https://github.com/Evil0ctal/Upper-Confidence-Bound-Pywebio.git
   cd Upper-Confidence-Bound-Pywebio
   ```

1. 使用 `pip` 安装所需依赖：

   ```bash
   pip install -r requirements.txt
   ```

### 依赖

该项目使用以下 Python 库：

- `pywebio` - 用于创建 Web 界面。
- `matplotlib` - 用于绘制图表。
- `numpy` - 用于数值计算。

你也可以手动安装依赖：

```bash
pip install pywebio matplotlib numpy
```

### 运行演示

1. 安装依赖后，使用以下命令运行演示：

   ```bash
   python main.py
   ```

2. 打开浏览器，并访问：

   ```bash
   http://localhost:8080
   ```

   你也可以使用提供的远程访问链接，从其他设备访问。

### 使用说明

- **UCB 算法模拟**：运行多轮 UCB 算法，观察其表现，以及如何在探索和利用之间取得平衡。
- **手动策略对比**：手动选择拉动臂，与 UCB 算法的表现进行对比。

### 许可证

此项目使用 MIT 许可证进行授权，详情请参见 LICENSE 文件。