# C45DecisionTreeForSk-learn
## 这是一个完美适配 sk-learn 接口的 C4.5 决策树实现，大家可以 copy 到自己的项目中直接使用。
## This is a C4.5 decision tree implementation perfectly adapted to the sk-learn interface, you can copy it into your project and use it directly.
#### C4.5 决策树基于信息增益比作为划分属性的标准，而不是 ID3 中使用的信息增益。我们注意到 sklearn 并未提供 C4.5 决策树的实现，为此，我通过继承 sklearn 中的 DecisionTreeClassifier 接口，实现了能够与 sklearn 接口完美适配的 C4.5 决策树。如果你需要，只需将 C4.5.py 直接加入你的项目中，无需担心适配问题。
#### C4.5 The decision tree is based on the information gain ratio as the criterion for dividing attributes, rather than the information gain used in ID3. We noticed that sklearn does not provide the implementation of the C4.5 decision tree. For this reason, I implemented the C4.5 decision tree that can perfectly adapt to the sklearn interface by inheriting the DecisionTreeClassifier interface in sklearn. If you need, just add C4.5.py directly to your project without worrying about adaptation.
