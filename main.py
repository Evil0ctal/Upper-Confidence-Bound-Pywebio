from pywebio import start_server
from pywebio.output import put_text, put_html, put_markdown, clear
from pywebio.input import input, FLOAT, actions
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
from pywebio.session import run_js


def ucb_demo():
    # 加载 MathJax，用于渲染 LaTeX 公式
    run_js("""
    window.MathJax = {
        tex: {inlineMath: [['$', '$'], ['\\(', '\\)']]}
    };
    """)
    put_html('<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>')

    # 设置 Matplotlib 字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

    put_markdown("# UCB 算法在线演示")

    # 添加算法讲解
    put_markdown(r"""
    ## 什么是 UCB 算法？

    UCB（Upper Confidence Bound，上置信界）算法是一种用于解决**多臂老虎机问题（Multi-Armed Bandit Problem）**的策略。该问题涉及在有限的选择中找到最佳选择，以最大化累积奖励。

    UCB 算法通过为每个选择计算一个“上置信界”来平衡**探索（Exploration）**和**利用（Exploitation）**。算法倾向于选择具有高平均奖励且探索次数较少的选项。

    **UCB 公式：**

    $$
    \text{UCB}_i = \bar{x}_i + \sqrt{\dfrac{2 \ln n}{n_i}}
    $$

    - $\bar{x}_i$：第 $i$ 个臂的平均奖励。
    - $n$：总拉动次数。
    - $n_i$：第 $i$ 个臂被拉动的次数。
    """)

    # 设置臂的数量
    num_arms = 5

    # 每个臂的真实中奖概率（算法未知）
    true_probs = np.random.rand(num_arms)

    put_text(f"共有 {num_arms} 个臂，每个臂的中奖概率未知。")

    rounds = input("请输入要模拟的轮数（建议 1000 以上）：", type=FLOAT)
    rounds = int(rounds)

    counts = np.zeros(num_arms)  # 每个臂被拉动的次数
    rewards = np.zeros(num_arms)  # 每个臂的累计奖励

    total_reward = 0
    cumulative_rewards = []
    selected_arms = []

    # 初始化：每个臂先拉一次
    for arm in range(num_arms):
        reward = np.random.binomial(1, true_probs[arm])
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        cumulative_rewards.append(total_reward)
        selected_arms.append(arm)

    # 运行 UCB 算法
    for t in range(num_arms, rounds):
        ucb_values = rewards / counts + np.sqrt(2 * np.log(t + 1) / counts)
        arm = np.argmax(ucb_values)
        reward = np.random.binomial(1, true_probs[arm])
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        cumulative_rewards.append(total_reward)
        selected_arms.append(arm)

    # 显示结果
    put_markdown("## 模拟结果")
    put_text(f"总奖励：{total_reward}")
    put_text(f"每个臂的真实中奖概率：{np.round(true_probs, 2)}")
    put_text(f"每个臂被选择的次数：{counts.astype(int)}")
    put_text(f"每个臂的估计中奖概率：{np.round(rewards / counts, 2)}")

    # 绘制累积奖励随时间的变化图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.xlabel('轮次')
    plt.ylabel('累积奖励')
    plt.title('累积奖励随时间的变化')

    # 绘制每个臂被选择的次数
    plt.subplot(1, 2, 2)
    plt.bar(range(num_arms), counts)
    plt.xlabel('臂编号')
    plt.ylabel('被选择次数')
    plt.title('每个臂被选择的次数')

    # 将图像转换为 base64 并显示
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    put_html(f'<img src="data:image/png;base64,{data}">')

    # 添加交互式演示
    put_markdown("## 交互式演示")

    put_text("您可以手动选择每一轮要拉动的臂，观察与 UCB 算法的差异。")
    manual_counts = np.zeros(num_arms)
    manual_rewards = np.zeros(num_arms)
    manual_total_reward = 0
    manual_cumulative_rewards = []

    for t in range(10):  # 限制交互轮数为10轮
        options = [{'label': f'拉动臂 {i}', 'value': i} for i in range(num_arms)]
        choice = actions(label=f"第 {t + 1} 轮：请选择要拉动的臂", buttons=options)
        selected_arm = choice
        reward = np.random.binomial(1, true_probs[selected_arm])
        manual_counts[selected_arm] += 1
        manual_rewards[selected_arm] += reward
        manual_total_reward += reward
        manual_cumulative_rewards.append(manual_total_reward)
        clear()
        put_markdown(f"### 第 {t + 1} 轮结果")
        put_text(f"您选择了臂 {selected_arm}，获得奖励：{reward}")
        put_text(f"当前总奖励：{manual_total_reward}")
        put_text(f"每个臂被选择的次数：{manual_counts.astype(int)}")
        estimated_probs = np.divide(manual_rewards, manual_counts, out=np.zeros_like(manual_rewards),
                                    where=manual_counts != 0)
        put_text(f"每个臂的估计中奖概率：{np.round(estimated_probs, 2)}")

    put_markdown("### 手动选择的累积奖励")
    plt.figure()
    plt.plot(manual_cumulative_rewards, label='手动策略')
    plt.plot(cumulative_rewards[:len(manual_cumulative_rewards)], label='UCB 算法')
    plt.xlabel('轮次')
    plt.ylabel('累积奖励')
    plt.title('手动策略与 UCB 算法的累积奖励对比')
    plt.legend()

    # 将图像转换为 base64 并显示
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    put_html(f'<img src="data:image/png;base64,{data}">')

    put_markdown("""
    ## 总结

    从以上结果可以看出，UCB 算法能够有效地在探索和利用之间取得平衡，逐渐倾向于选择实际中奖概率较高的臂，从而获得更高的累积奖励。

    在手动选择中，如果没有策略或经验，很难达到 UCB 算法的效果。这也体现了 UCB 算法在解决多臂老虎机问题上的优势。
    """)


if __name__ == '__main__':
    start_server(ucb_demo, port=8080, remote_access=True)
