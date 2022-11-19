# 深度學習理論——VC Bound, Pac-Bayesian Framework 和 Neural Tangent Kernel
> [time=Fri, Nov 18, 2022 12:06 AM]
> [name=主講: Mark Chang 老師 (MediaTek Taiwan AI Researcher)]

在建立預測模型的時候，我們會將資料分為測試集(training data)、訓練集(testing data)，預測結果分別算出training error 和testing error，理想的情況是兩者都很小，而如果是training error 很小，但testing error 很大的情況，我們會稱作**overfitting**，此時會思考是否為在模型中放入過多參數(parameters)而導致overfitting。
<br>
## VC Bound——如何衡量是否overfitting

### VC Bound
$$\epsilon(h)\le\hat{\epsilon(h)}+\sqrt{\tfrac{8}{n}log(\tfrac{4(2n)^d}{\delta})}$$
不等式左邊為testing error，右邊為training error 加上開根號，其中開根號內有可以表達data 數量的n 和代表模型複雜度的d，因此如果training error 很小，加上資料夠多、模型複雜度不要太大，如此一來可以確定testing error 是小的。

### 何謂VC Dimension?
簡要來說VC Dimension 跟模型複雜度有關。
* 線性模型的VC Dimension：
通常是有幾個參數，VC Dimension 就是多少
$$O(W)$$$$W = number of parameters$$
* 全連接層的神經網絡VC Dimension：
$$O(LW log W)$$$$L = number of layers$$$$W = number of parameters$$

由上述VC Bound 公式可以找出最適合的VC Dimension，由下圖可知，藍色線為training error 會隨著參數愈多（VC Dimension 愈大），不過超過一個程度會使testing error（綠色線）開始上升，因此可以找到testing error 最小的一個參數量，即為VC Dimension 最佳的點，再過去反而會overfitting。
<br>
![](https://i.imgur.com/E2XGdx6.png)
<br>
上圖為機器學習模型的概念，那在深度學習模型，其實是參數放愈多，testing error 反而會掉下來，模型效果會愈好，稱為**over-parameterization**，以下圖為示意圖，來自*Reconciling modern machine learning and the bias-variance trade-off* (Mikhail Belkin，2018)此篇論文的研究結果。也因此在資料、變數都很多的實務上，會傾向使用深度學習模型。
<br>
![](https://i.imgur.com/cQDGcSP.png)
<br>
另外，在深度學習中，模型會不會overfitting 並不會跟參數量有直接關係，而是feature(input) 跟label 要有關聯性，才會影響模型預測的好壞。

總結：Testing error 會受資料分佈影響；而VC Bound 則跟資料分布無關。
<br>

## Pac-Bayesian Framework

 一般做深度學習要減小testing error 所使用的regularization 方法可能會是weight decay 或data augmentation 的方式，但通常改善可能不會很明顯，而若我們套入VC Bound 改善可能會很明顯，因此這裡提出Pac-Bayesian Framework。



首先介紹以下兩種模型：
* Deterministic Model
指用單一的假設去預測。
![](https://i.imgur.com/zH1Ok9Q.png)
<br>
* Stochastic Model (Gibbs Classifier)
有一整群的假設，一次從中抽取一個假設去算，做很多次後把所有得到的testing error 取期望值。
<br>
![](https://i.imgur.com/rwJQODo.png)
<br>

Pac-Bayesian Bound 就是以Stochastic Model 的觀點來看，PAC-Bayesian 可以將陡峭的minimum 考慮進去，他會考慮周圍的部分，才不會選到陡峭的minimum，而是選較平滑的那個min(才不會over-fitting)。
<br>
![](https://i.imgur.com/WZ7mybq.png)
<br>
<br>
### PAC-Bayesian Bound 公式
![](https://i.imgur.com/x7wZFxK.png)
跟VC Bound 很大的不同在於有KL divergence： $$KL(Q||P)$$ ，其中P 為訓練前的模型，Q 為訓練後的模型，若在訓練過程中模型改變很多的話，testing error 和training error 差距會比較大，容易造成overfitting，因此此觀點思考深度模型為何overfitting 不只從模型複雜度來看，而是考慮到訓練前後模型的改變程度。

PAC-Bayesian Bound 特性是他是**data-dependent**，不管模型複雜度高低都不會影響，是跟資料乾淨與否比較有關，資料愈乾淨，PAC-Bayesian Bound 愈小。
> 【比較】
> -VC Dim 大小與模型複雜度比較有關
> -PAC-Bayesian 大小主要跟資料乾淨與否有關、也跟一點點模型複雜度有關

## Neural Tangent Kernel
此方法可以解釋深度學習模型為什麼可以不會over-fitting，以及為什麼training error 可以達0。


