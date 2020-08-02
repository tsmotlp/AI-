# 线性模型

##  1. 线性回归

假设有数据 $D=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{n},y_{n})\}=\{(x_{i},y_{i})\}_{i=1}^{n}$，其中 $x_{i}\in\mathbb{R^{p}}$，$y_{i}\in\mathbb{R}$，$i=1,\cdots,n$. 用矩阵的形式可以表示成：
$$
X=(x_{1},x_{2},\cdots,x_{n})^{T}=\begin{bmatrix} x_{1}^{T}\\ x_{2}^{T}\\ \vdots\\ x_{n}^{T}\end{bmatrix}=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p}\\ x_{21} & x_{22} & \cdots & x_{2p}\\ \vdots & \vdots & \ddots & \vdots\\ x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix}_{n \times p}
$$

$$
Y=(y_{1},y_{2},\cdots,y_{n})^{T}=\begin{bmatrix} y_{1}\\ y_{2}\\ \vdots\\ y_{n}\end{bmatrix}_{n \times 1}
$$

此时我们需要找到这堆数据的规律，或者说找一个函数来拟合这些数据。我们假设这个函数是 $f(x)=w^{T} \cdot x+b$，即对于每个$x_{i}$，都有：
$$
\begin{cases}
\hat y_{1}=w_{1} \cdot x_{11}+w_{2} \cdot x_{12}+ \cdots + w_{p} \cdot x_{1p}+b\\
\hat y_{2}=w_{1} \cdot x_{21}+w_{2} \cdot x_{22}+ \cdots + w_{p} \cdot x_{2p}+b\\
\cdots\\
\hat y_{n}=w_{1} \cdot x_{n1}+w_{2} \cdot x_{n2}+ \cdots + w_{p} \cdot x_{np}+b
\end{cases}
$$
写成矩阵形式就是：
$$
\begin{bmatrix}\hat y_{1}\\\hat y_{2}\\ \vdots\\ \hat y_{n}\end{bmatrix}_{n \times 1}= \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p}\\ x_{21} & x_{22} & \cdots & x_{2p}\\ \vdots & \vdots & \ddots & \vdots\\ x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix}_{n \times p} \cdot \begin{bmatrix} w_{1}\\ w_{2}\\ \vdots\\ w_{p}\end{bmatrix}_{p \times 1} +\begin{bmatrix} b\\ b\\ \vdots\\ b\end{bmatrix}_{n \times 1}
$$

$$
\hat Y=X \cdot W+b
$$

上式也可以写成：
$$
\hat Y=X \cdot W+b \cdot 1
$$
为了统一，我们可以令 $w_{0}=b, x_{i0}=1$，则有：
$$
\begin{bmatrix}\hat y_{1}\\\hat y_{2}\\ \vdots\\ \hat y_{n}\end{bmatrix}_{n \times 1}= \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p}\\ \vdots & \vdots & \ddots & \vdots & \vdots\\ 1 & x_{n1} & x_{n2} & \cdots & x_{np}\end{bmatrix}_{n \times (p+1)} \cdot \begin{bmatrix}b\\ w_{1}\\ w_{2}\\ \vdots\\ w_{p}\end{bmatrix}_{(p+1) \times 1}
$$
则可以统一成如下形式：
$$
\hat Y=X \cdot W
$$
其中$X=(x_{1},x_{2}, \cdots, x_{n})^{T},W=(b, w_{1},w_{2}, \cdots , w_{p}),而x_{i}=(1, x_{i1},x_{i2}, \cdots , x_{ip})^{T}$.
那么现在的主要问题就变成怎么求得一个最好的 $W$，使我们假设的这个函数尽可能好的拟合原本的数据，这个问题的关键在于如何衡 $\hat Y$ 与 $Y$ 之间的差别。有很多种方法可以实现这个差别的度量，我们在这里使用最常见的一种性能度量方法——均方误差(Mean Square Error):
$$
\begin{aligned}
E_{W}&=\|\hat Y-Y\|_{2}=\|XW-Y\|_{2} \\ &=(XW-Y)^{T} \cdot (XW-Y)\\&=(W^{T}X^{T}-Y^{T}) \cdot (XW-Y)\\&=W^{T}X^{T}XW-W^{T}X^{T}Y-Y^{T}XW+Y^{T}Y\\&=W^{T}X^{T}XW-2W^{T}X^{T}Y+Y^{T}Y
\end{aligned}
$$
在不考虑过拟合的情况下，均方误差越小，就说明我们的找到的函数越能够代表原本数据的分布规律。基于均方误差最小化来进行模型求解的方法称为“最小二乘法”(Least Square Method)，即将 $E_{W}$ 对 $W$ 求导，得到：
$$
\begin{aligned}
\frac{\partial E_{W}}{\partial W}&=\frac{\partial }{\partial W}(W^{T}X^{T}XW-2W^{T}X^{T}Y+Y^{T}Y)\\&=2X^{T}XW-2X^{T}Y
\end{aligned}
$$
令其等于0可以得到：
$$
\begin{aligned}
&2X^{T}XW-2X^{T}Y=0\\
\Rightarrow&X^{T}XW=X^{T}Y\\
\Rightarrow&W=(X^{T}X)^{-1}X^{T}Y
\end{aligned}
$$
其中，我们把 $(X^{T}X)^{-1}X^{T}$ 称之为 $X$ 的伪逆(Moore-Penrose)，记为$X^{+}=(X^{T}X)^{-1}X^{T}$。到此，我们就可以得到：
$$
W^{*}= \arg \min_{W}E_{W}=(X^{T}X)^{-1}X^{T}Y
$$
即$w^{*}=(W^{*}_{1},W^{*}_{2}, \cdots,W^{*}_{p})^{T}$，$b^{*}=W^{*}_{0}。$则最优的拟合函数为：
$$
y=w^{*T} \cdot x+b^{*}
$$

##  2. 广义线性模型

在线性回归模型中，我们其实是用 $w^{T} \cdot x+b$ 去逼近样本的真实标记 $y$，那么如果我们不是去逼近 $y$，而是去逼近它的衍生物呢？比如说用 $w^{T} \cdot x+b$ 去逼 $\ln y$，即：
$$
\ln y = w^{T} \cdot x+b\\
y=e^{w^{T} \cdot x+b}
$$
此时该式在形式上仍是线性回归，但是实质上已经是在求从输入空间到输出空间的非线性映射。更一般的，考虑一个单调可微函数 $g(\cdot)$，令：
$$
y = g^{-1}(w^{T} \cdot x+b)
$$
我们称这个模型为广义线性模型，其中单调可微函数 $g(\cdot)$ 称为联系函数。

$\textbf{参考资料}：$

  1. 周志华《机器学习》
  2. [机器学习-白板推导系列](https://www.bilibili.com/video/av70839977?p=9)
