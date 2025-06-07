#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from torch import nn, optim
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


plt.rcParams ['font.sans-serif'] ='SimHei'               #显示中文
plt.rcParams ['axes.unicode_minus']=False               #显示负号


# ## 读取数据查看信息

df=pd.read_csv('Clean Data_pakwheels.csv')
df=df[[col for col in df.columns if 'Unnamed' not in col]]
df.head(2)


df.info()


# |英文列名称|	中文列名称|	描述
# |-----------|----------|--------
# |Company Name|	公司名称|	汽车制造商的名称。
# |Model Name	|型号名称|	汽车的型号。
# |Price	|价格	|汽车的售价。
# |Model Year|	模型年份	|汽车制造的年份。
# |Location	|位置	|汽车的所在地或销售地。
# |Mileage	|里程	|汽车已经行驶的总里程数。
# |Engine Type	|引擎类型	|汽车的引擎类型（如柴油、汽油）。
# |Engine Capacity	|引擎容量|	汽车引擎的容量或大小。
# |Color	|颜色	|汽车的颜色。
# |Assembly	|组装	|汽车是在本地组装还是进口组装。
# |Body Type	|车身类型	|汽车的车身形状（如轿车、SUV）。
# |Transmission Type	|变速器类型	|汽车的变速器类型（如自动、手动）。
# |Registration Status	|注册状态	|汽车的注册情况（如已注册、未注册）。


#观察缺失值
import missingno as msno
msno.matrix(df)


# ## 没有缺失值，直接开始特征工程


#查看数值型数据,
pd.set_option('display.max_columns', 30)
df.select_dtypes(exclude=['object']).head()


# Price作为响应变量y，Model Year转为车龄age，其他变量不用处理


y=df['Price']; df['age']=2024-df['Model Year']  



#查看非数值型数据
df.select_dtypes(exclude=['int64','float64']).head()




for col in df.select_dtypes(exclude=['int64','float64']).columns:
    print(f'{col}特征类别数量为{len(df[col].unique())}')


# 其中，
# - Model Name特征变量类别太多，进行删除。
# - 颜色24类有很多类似的颜色可以进行归类，减少变量。
#     - Silver/Grey: 包括银色和灰色。
#     - White/Beige: 包括白色和米色。
#     - Black: 只包括黑色。
#     - Brown系: 包括棕色、栗色、褐红色和铜色。
#     - Gold: 只包括金色。
#     - Blue系: 包括蓝色、海军蓝和靛蓝。
#     - Red系: 包括红色、粉色、洋红和酒红色。
#     - Green系: 包括绿色和绿松石色。
#     - Orange: 只包括橙色。
#     - Purple: 只包括紫色。
#     - Yellow: 只包括黄色。
#     - Unlisted: 包括未列出的颜色。
# - Company Name特征类别数量为31，也要进行适当的归类处理



df=df.drop(columns=['Model Name'])



color_mapping = {
    'Silver': ['Silver', 'Grey'],
    'White': ['White', 'Beige'],
    'Black': ['Black'],
    'Brown': ['Brown', 'Maroon', 'Burgundy', 'Bronze'],
    'Gold': ['Gold'],
    'Blue': ['Blue', 'Navy', 'Indigo'],
    'Red': ['Red', 'Pink', 'Magenta', 'Wine'],
    'Green': ['Green', 'Turquoise'],
    'Orange': ['Orange'],
    'Purple': ['Purple'],
    'Yellow': ['Yellow'],
    'Unlisted': ['Unlisted', 'Assembly']  
}
inverse_color_mapping = {val: key for key, vals in color_mapping.items() for val in vals}
# 映射
df['Color'] = df['Color'].map(inverse_color_mapping)
print(len(df['Color'].unique()))



### 等同下面这个方法
inverse_color_mapping = {}

# 对原始字典进行双重循环
for main_color, variations in color_mapping.items():
    for color in variations:
        # 将变体颜色作为键，主要颜色作为值添加到字典中
        inverse_color_mapping[color] = main_color
inverse_color_mapping


# #### 颜色类别减少为12，下面对汽车品牌处理



c=df['Company Name'].value_counts()  # 统计每个品牌车辆的样本数量，然后计算每个品牌车辆的均价
df.groupby(['Company Name']).mean(numeric_only=True).loc[c.index,:].assign(count=c.to_numpy())[['Price','count']].style.bar(align='mid', color=['#491256', 'skyblue'])


# Suzuki，Toyota，Honda，Daihatsu四个品牌已经大概包含了大部分车辆，其他的剩下面的车辆可以归为一类，
# 但是有些牌子可能是奢侈品豪车，均价超级贵，所以我们把下面的Price均价分为三个区间，0-100w的作为普通其他类，100w-500w作为中档其他类，500w以上为高档其他类



c_p=df.groupby(['Company Name']).mean(numeric_only=True).loc[c.index,:]['Price'].iloc[4:]
print(f'普通其他类:{c_p[c_p<1000000].index}')
print(f'中档其他类:{c_p[(c_p > 1000000) & (c_p < 5000000)].index}')
print(f'高档其他类:{c_p[c_p>5000000].index}')




brand_mapping = {
    'Suzuki': 'Suzuki',
    'Toyota': 'Toyota',
    'Honda': 'Honda',
    'Daihatsu': 'Daihatsu',   #这四个牌子保留
    'General': ['Hyundai', 'United', 'Daewoo', 'Chevrolet', 'Chery', 'Fiat', 'Adam'],
    'Mid-Range': ['Nissan', 'Mitsubishi', 'FAW', 'Mazda', 'KIA', 'Subaru', 'SsangYong', 'Land', 'DFSK', 'Jeep', 'MINI', 'Volvo'],
    'Premium': ['Mercedes', 'Audi', 'BMW', 'Lexus', 'Range', 'Porsche', 'Hummer', 'Jaguar'] }
# Inverting the mapping
inverse_brand_mapping = {val: key for key, vals in brand_mapping.items() for val in ([vals] if isinstance(vals, str) else vals)}
df['Company Name'] = df['Company Name'].map(inverse_brand_mapping, na_action='ignore')  # na_action='ignore' to keep the original value if not found in mapping
print(len(df['Company Name'].unique()))



### 等同下面这个方法
inverse_brand_mapping = {}

# 对原始字典进行双重循环
for brand, models in brand_mapping.items():
    # 检查models是否是字符串类型，如果是，将其转换为列表
    if isinstance(models, str):
        models = [models]
    # 遍历模型列表
    for model in models:
        # 将牌子作为键，品牌作为值添加到字典中
        inverse_brand_mapping[model] = brand
inverse_brand_mapping




for col in df.select_dtypes(exclude=['int64','float64']).columns:
    print(f'{col}特征类别数量为{len(df[col].unique())}')




df=df.drop(columns=['Price'])


# ## 数据画图探索

# ### 数值型变量画图



#查看特征变量的箱线图分布
num_columns = df.select_dtypes(exclude=['object']).columns.tolist() # 列表头
dis_cols = 2                   #一行几个
dis_rows = len(num_columns)
plt.figure(figsize=(3 * dis_cols, 2.5 * dis_rows),dpi=128)
 
for i in range(len(num_columns)):
    plt.subplot(dis_rows,dis_cols,i+1)
    sns.boxplot(data=df[num_columns[i]], orient="v",width=0.5)
    plt.xlabel(num_columns[i],fontsize = 16)
plt.tight_layout()
#plt.savefig('特征变量箱线图',formate='png',dpi=500)
plt.show()


# 车龄存在一些极大异常值，有一些车车龄很大。Mileage ，engine capacity 也有很多极大值

# 型号年份（Model Year）：大部分车辆的型号年份集中在2005年到2015年之间，中位数大约在2010年左右。这表明大多数车辆相对较新。箱线图下方的长尾表明也有少量较旧的车辆，但这些较老的车辆相对不多。
# 
# 里程（Mileage）：大多数车辆的里程较低，中位数接近0，但范围广泛，上至100万公里。这说明有些车辆使用非常频繁，但大多数车辆保持较低的里程数。长尾分布表明有些车辆的里程异常高，这可能反映了部分车辆经受了长期的使用。
# 
# 引擎容量（Engine Capacity）：大部分车辆的引擎容量在0到2000cc之间，中位数在1000cc到1500cc左右。这表明大多数车辆装配的是中小型引擎。存在一些异常点，表明有少量车辆装配了非常小或非常大的引擎。
# 
# 车龄（Age）：车龄的分布显示，大多数车辆的年龄集中在5到15年之间，中位数约为10年。这意味着大多数车辆都不是很新，但也未达到极老的状态。和型号年份的分析相符，说明市场上流通的车辆大多数都有一定的使用历史。



#画密度图，训练集和测试集对比
dis_cols = 2                   #一行几个
dis_rows = len(num_columns)
plt.figure(figsize=(3 * dis_cols, 2 * dis_rows),dpi=256)
 
for i in range(len(num_columns)):
    ax = plt.subplot(dis_rows, dis_cols, i+1)
    ax = sns.kdeplot(df[num_columns[i]], color="skyblue" ,fill=True)
    ax.set_xlabel(num_columns[i],fontsize = 14)
plt.tight_layout()
#plt.savefig('训练测试特征变量核密度图',formate='png',dpi=500)
plt.show()


# 和上面结论一致，age ， Mileage ，engine capacity ，典型的右偏分布 存在很多极大值

# ### 分类型变量画图



df.select_dtypes(exclude=['int64','float64']).columns




# Select non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
# Set up the matplotlib figure with 2x4 subplots
f, axes = plt.subplots(4, 2, figsize=(10,12),dpi=128)
# Flatten axes for easy iterating
axes_flat = axes.flatten()
# Loop through the non-numeric columns and create a countplot for each
for i, column in enumerate(non_numeric_columns):
    if i < 8:  # Check to avoid IndexError if there are more than 8 non-numeric columns
        sns.countplot(x=column, data=df, ax=axes_flat[i])
        axes_flat[i].set_title(f'Count of {column}')
        for label in axes_flat[i].get_xticklabels():
            label.set_rotation(45)

# Hide any unused subplots
for j in range(i + 1, 8):
    f.delaxes(axes_flat[j])
plt.tight_layout()
plt.show()


# 公司名称：图表显示Toyota和Suzuki是数量最多的品牌，其次是Honda。这可能表明在相关市场中，Toyota和Suzuki更受欢迎，可能因为它们的可靠性、品牌认知度或性价比。Premium（可能代表高端品牌）和General（可能代表普通或非专门品牌）的数量明显少于前三者，这表明市场可能更倾向于具有中等价格和质量的车辆。
# 
# 地点：该图展示了汽车的地理分布情况。伊斯兰堡（Islamabad）的数量最多，其次是旁遮普省（Punjab）。这可能反映了这些区域的高车辆需求或更高的购买力。
# 
# 引擎类型：绝大多数车辆使用的是汽油（Petrol）引擎，而柴油（Diesel）和混合动力（Hybrid）引擎的车辆数量相对较少。这可能是由于汽油车的普及度高，以及在某些地区对于柴油和混合动力车的支持不足。
# 
# 颜色：银色（Silver）和白色（White）车辆在数量上占据优势，这可能因为这些颜色的车辆更受欢迎，也可能与热量反射能力或经典美观有关。未上市（Unlisted）颜色的数量相对较少，可能包含特殊或定制的颜色。
# 
# 装配类型：本地装配（Local）的车辆远多于进口车辆（Imported）。这可能反映了本地制造车辆的成本优势或进口税的影响。
# 
# 车身类型：轿车（Sedan）和掀背车（Hatchback）是最常见的车身类型，而SUV的数量相对较少。这可能反映了市场上对于日常使用和经济型车辆的高需求。
# 
# 变速器类型：自动档（Automatic）车辆数量少于手动档（Manual）车辆，这可能与成本、驾驶习惯或车辆类型有关。
# 
# 注册状态：已注册（Registered）的车辆数量远超未注册（Un-Registered）的车辆，这显示了大多数车辆在使用前已完成了注册流程。

# ### 响应变量y的分布



# 查看y的分布
#回归问题
plt.figure(figsize=(7,3),dpi=128)
plt.subplot(1,3,1)
y.plot.box(title='响应变量箱线图')
plt.subplot(1,3,2)
y.plot.hist(title='响应变量直方图')
plt.subplot(1,3,3)
y.plot.kde(title='响应变量核密度图')
#sns.kdeplot(y, color='Red', shade=True)
#plt.savefig('响应变量.png')
plt.tight_layout()
plt.show()


# 有些车的价格异常值太夸张，处理一些



#处理y的异常值  二千万以上的车就去掉
y=y[y <= 20000000]
plt.figure(figsize=(7,3),dpi=128)
plt.subplot(1,3,1)
y.plot.box(title='响应变量箱线图')
plt.subplot(1,3,2)
y.plot.hist(title='响应变量直方图')
plt.subplot(1,3,3)
y.plot.kde(title='响应变量核密度图')
#sns.kdeplot(y, color='Red', shade=True)
#plt.savefig('响应变量.png')
plt.tight_layout()
plt.show()




#筛选给x
df=df.iloc[y.index,:]
print(f'删除了样本数量：{46022-df.shape[0]}')
df.shape


# ### X异常值处理


print(df)
#X变量独热处理
X=pd.get_dummies(df)
#X异常值处理，先标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s = scaler.fit_transform(X)


# 然后画图查看



plt.figure(figsize=(20,8),dpi=128)
plt.boxplot(x=X_s,labels=X.columns)
plt.hlines([-10,10],0,len(X.columns))
plt.xticks(rotation=90)
plt.show()


# 样本个体数据方大于整体数据10倍的标准差之外就筛掉



#异常值多的列进行处理
def deal_outline(data,col,n):   #数据，要处理的列名，几倍的方差
    for c in col:
        mean=data[c].mean()
        std=data[c].std()
        data=data[(data[c]>mean-n*std)&(data[c]<mean+n*std)]
        #print(data.shape)
    return data




X=deal_outline(X,X.columns,10)  #筛掉异常值

print(X.iloc[1])
print(X)
# In[29]:


#取值唯一的独热变量删除
for col in X.columns:
    if len(X[col].value_counts())==1:
        #print(X[col])
        print(col)
        X.drop(col,axis=1,inplace=True)
y=y[X.index]
print(f'删除了样本数量：{df.shape[0]-X.shape[0]}')
X.shape,y.shape


# ### 相关系数矩阵



#变量独热之后有点多，就不把颜色和车身类型等变量放上去了
X_plot=X[[col for col in X.columns if 'Color' not in col]]
X_plot=X_plot[[col for col in X_plot.columns if 'Body' not in col]]



corr = plt.subplots(figsize = (18,16),dpi=128)
corr= sns.heatmap(X_plot.assign(Y=y).corr(method='spearman'),annot=True,square=True)


# 明显的负相关关系：一些变量对显示为深紫色，表示它们之间存在较强的负相关。例如，“Mileage”和“Engine Capacity”之间似乎有一定的负相关性，这可能意味着发动机容量越大，车辆的里程数越少，这是有意义的，因为大发动机通常消耗更多的燃料。
# 
# 明显的正相关关系：一些变量对显示为深红色，指示它们之间存在强烈的正相关。例如，“Engine Type_Petrol”和“Company Name_General”可能有较高的正相关，这可能意味着某个特定的汽车制造商更倾向于生产汽油发动机车型。
# 
# 弱相关或无相关关系：很多变量对显示为接近黑色，表示它们之间几乎没有或没有明显的相关性。这表明很多车辆特征之间可能没有直接的或显著的影响关系。
# 
# 特定地区或特征的相关性：可以注意到某些特定区域（如Location_JK, Location_Punjab等）和其他变量之间存在不同程度的相关性。这可能表明地区因素对车辆的某些属性有影响，例如某一地区可能更倾向于特定类型的车辆或发动机类型。

# ## 开始机器学习！



#划分训练集和验证集和测试集
from sklearn.model_selection import train_test_split
#X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.model_selection import train_test_split

# 首先分割出训练集和临时集（包含验证和测试）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% 训练，30% 临时

# 然后从临时集中分割出验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)  # 50% 验证，50% 测试



#数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
print('训练数据形状：')
print(X_train_s.shape,y_train.shape)
print('验证数据形状：')
print(X_val_s.shape,y_val.shape)

config = {
    'epoch': 100,
    'batch_size': 512,
    'learning_rate': 0.1,
    'device': 'cpu'
}

# 定义网络结构
class Network(nn.Module):
    def __init__(self, in_dim, hidden_1, hidden_2, hidden_3, hidden_4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Linear(hidden_3, hidden_4),
            nn.BatchNorm1d(hidden_4),
            nn.ReLU(),
            nn.Linear(hidden_4, 1)
        )

    def forward(self, x):
        y = self.layers(x)
        return y

# 定义网络
model = Network(X_train_s.shape[1], 256, 256, 256, 32)
# 网络的输入维度由 train_data.shape[1] 确定，即训练数据的特征数量。其余参数分别为隐藏层的大小，分别是 256、256、256 和 32。
model.to(config['device'])

# 使用Xavier初始化权重
for line in model.layers:
    if type(line) == nn.Linear:
        print(line)
        nn.init.xavier_uniform_(line.weight)

# 将数据转化为tensor，并移动到cpu或cuda上

train_features = torch.tensor(X_train_s, dtype=torch.float32, device=config['device'])
train_num = train_features.shape[0]
train_labels = torch.tensor(y_train, dtype=torch.float32, device=config['device'])

validation_features = torch.tensor(X_val_s, dtype=torch.float32, device=config['device'])
validation_num = validation_features.shape[0]
validation_labels = torch.tensor(y_val.values, dtype=torch.float32, device=config['device'])



# 定义损失函数和优化器
criterion = nn.MSELoss()
criterion.to(config['device'])
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 开始训练

mae_list = []

for epoch in range(config['epoch']):
    losses = []
    # 将模型设置为训练模式。在训练模式下，如 Batch Normalization 和 Dropout 等层的行为会有所不同。
    model.train()
    # mini-batch 循环：
    for i in range(0, train_num, config['batch_size']):
        end = i + config['batch_size']
        if i + config['batch_size'] > train_num - 1:
            end = train_num - 1
        mini_batch = train_features[i: end]
        mini_batch_label = train_labels[i: end]
        # 使用模型对当前 mini-batch 进行前向传播，得到预测值。
        pred = model(mini_batch)
        # 移除预测张量中的大小为 1 的维度。计算损失
        pred = pred.squeeze()
        loss = criterion(pred, mini_batch_label)
        # 检查损失是否为 NaN，如果是，则停止当前 epoch 的训练。
        if torch.isnan(loss):
            break
        # 计算平均绝对误差（Mean Absolute Error, MAE），然后将结果添加到 losses 列表中。
        mae = torch.abs(mini_batch_label - pred).sum() / (end - i)
        mae2 = torch.abs(mini_batch_label - pred).mean().item()
        losses.append(mae.item())
        # 清零优化器的梯度缓存。
        optimizer.zero_grad()
        # 计算损失相对于模型参数的梯度。
        loss.backward()
        # 根据计算出的梯度更新模型参数。
        optimizer.step()
    # 将模型设置为评估模式，在评估模式下，Batch Normalization 和 Dropout 等层的行为会有所不同。
    model.eval()
    # 使用模型对验证集进行前向传播，得到预测值。
    pred = model(validation_features)
    validation_mae = torch.abs(validation_labels - pred.squeeze()).sum().item() / validation_num

    mae_list.append((sum(losses) / len(losses), validation_mae))

    print(f"epoch:{epoch + 1} MAE: {sum(losses) / len(losses)}, Validation_MAE: {validation_mae}")
    print(mae2)
    torch.save(model, 'model.pth')


# 加载模型
model = torch.load('model.pth', weights_only=False)
model.eval()  # 设置为评估模式

#数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test)
X_test_s = scaler.transform(X_test)

# 确保 test_data 是张量
test_data = torch.tensor(X_test_s, dtype=torch.float32, device=config['device'])

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    predictions = model(test_data)

# 如果需要，去掉多余的维度
predictions = predictions.squeeze()

# 将实际标签转换为张量（如果尚未转换）
actual_labels = torch.tensor(y_test.values, dtype=torch.float32, device=config['device'])

# 计算 MAE
mae = torch.abs(actual_labels - predictions).mean().item()

# 打印预测结果和 MAE
print(f"MAE: {mae}")


