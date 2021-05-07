<script type="text/x-mathjax-config">   MathJax.Hub.Config({     tex2jax: {       inlineMath: [ ['$','$'], ["\\(","\\)"] ],       processEscapes: true     }   }); </script>

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 2 Basics of Neural Network programming

## Logistic Regression
<div align=center>
<img src="./images/logistic-regression.png" alt="Logistic Regression" width=600 />
</div>

给定输入 $\vec{x}\in\mathbb{R}^{n_x}$，想要预测得到$\hat{y}=P(y=1|\vec{x})$

两个参数 $w\in\mathbb{R}^{n_x}, b\in\mathbb{R}$

由于要使得$\hat{y}\in[0,1]$，所以加上一个sigmoid函数$\sigma(z)=\frac{1}{1+e^{-z}}$

所以输出
$$
\hat{y}=\sigma(w^Tx+b)
$$


## Logistic Regression Cost Function

<img src="./images/logistic-regression-cost-function.png" alt="Logistic Regression Cost Function" style="width:500px;" />

Loss function是单个的

Cost function是加起来求和

这里的Cost Function是凸函数



## Gradient Descent

<img src="./images/gradient-descent-1.png" alt="Gradient Descent 1" style="width:500px;" />

<img src="./images/gradient-descent-2.png" alt="Gradient Descent 1" style="width:500px;" />
$$
w := w-\alpha\frac{\partial J(w,b)}{\partial w} \\
b := w-\alpha\frac{\partial J(w,b)}{\partial b}
$$

## Computing Derivatives with a Computational Graph

反向传播 —> 链式法则

编程时用`dvar`指代$\frac{\partial{J}}{\partial{var}}$



## Logistic Regression Gradient Descent

<img src="./images/logistic-regression-derivatives-1.png" alt="Logistic Regression Derivatives 1" style="width:500px;" />

<img src="./images/logistic-regression-derivatives-2.png" alt="Logistic Regression Derivatives 2" style="width:500px;" />
$$
da=\frac{\partial{J}}{\partial{a}}=-\frac{y}{a}+\frac{1-y}{1-a} \\
\frac{\partial{a}}{\partial{z}}=\frac{e^{-z}}{(1+e^{-z})^2}=a(1-a) \\
dz=\frac{\partial{J}}{\partial{z}}
=\frac{\partial{J}}{\partial{a}}\frac{\partial{a}}{\partial{z}}
=(-\frac{y}{a}+\frac{1-y}{1-a})*a(1-a)=a-y \\
$$
所以
$$
z=w_1x_1+w_2x_2+b \\
dw_1=x_1dz=x_1(a-y) \\
dw_2=x_2dz=x_2(a-y) \\
db=dz=a-y
$$
当有 m 个Examples时

<img src="./images/logistic-regression-on-m-examples-1.png" alt="Logistic Regression on m Examples 1" style="zoom:50%;" />

<img src="./images/logistic-regression-on-m-examples-2.png" alt="Logistic Regression on m Examples 2" style="zoom:50%;" />

m个Examples时，每次计算就是对每个样本$x^i$求得得$dw$求和，最后取平均算出最终的$dw$，然后开始迭代$w$

这样的缺陷是，会进行两层循环，$m * n$ 次
