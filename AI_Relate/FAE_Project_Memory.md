# FAE Project Memory

## 1. 项目一句话定位

FAE（FeAture Explorer）是一个基于 **Python + PySide6** 的桌面端放射组学/医学分析工具，围绕三条主线提供 GUI 工作流：

- `Feature`：影像特征提取、特征合并、ICC 评估
- `BC`：二分类建模（Binary Classification）
- `SA`：生存分析（Survival Analysis）

它不是 Web 项目，也不是标准 Python 包，更像一个面向研究场景的 GUI 应用集合。

---

## 2. 当前代码库最重要的全局认知

### 2.1 启动模型

主入口有两个：

- `MainFrameCall.py`
- `MainFrameCall_opt.py`

两者都负责：

- 判断当前是源码运行还是 PyInstaller 打包运行
- 将工作目录切换到程序所在目录
- Windows 下启用 legacy filesystem encoding
- 创建 `QApplication`
- 启动首页窗体 `HomeUI\HomePageForm.py`

可记为：

`MainFrameCall*.py` -> `HomeUI\HomePageForm.py` -> 各业务子窗体

其中：

- `MainFrameCall.py` 是标准入口
- `MainFrameCall_opt.py` 额外调用了 `QApplication.setAttribute(Qt.AA_DontUseNativeDialogs)`，可视为一个“优化/兼容模式”入口

### 2.2 首页就是总导航器

`HomeUI\HomePageForm.py` 是整个产品的导航中枢。

它在初始化时会直接实例化多个子模块窗体，然后通过按钮点击执行 `show()` / `hide()` 切换。当前首页直接挂接的模块有：

- Feature Extraction
- Feature Merge
- ICC Estimation
- BC Preprocessing
- BC Model Exploration
- BC Visualization
- BC Prediction
- SA Model Exploration
- SA Visualization
- Plugin Run

所以当前架构不是“懒加载路由式页面系统”，而是：

**首页预先持有多个子模块实例的桌面应用结构**

这对快速开发 GUI 很直接，但对解耦、测试、模块边界不太友好。

---

## 3. 顶层目录结构速记

仓库根目录下最重要的业务目录如下：

### `HomeUI\`

首页和产品入口相关内容。

关键文件：

- `HomeUI\HomePage.ui`：Qt Designer 原始界面
- `HomeUI\HomePage.py`：由 `.ui` 转换得到的界面类
- `HomeUI\HomePageForm.py`：首页逻辑控制器
- `HomeUI\VersionConstant.py`：版本常量

当前版本信息：

- `MAJOR = 0`
- `MINOR = 7`
- `PATCH = 0`
- `VERSION = 0.7.0`
- `ACCEPT_VERSION = ['0.7.0']`

### `Feature\`

放射组学特征工程模块。

当前可确认职责：

- 图像与 ROI 文件匹配
- 序列匹配
- Radiomics 参数配置
- 特征提取
- 特征合并
- ICC 估计

关键文件：

- `Feature\FileMatcher.py`
- `Feature\SeriesMatcher.py`
- `Feature\RadiomicsParamsConfig.py`
- `Feature\GUI\FeatureExtractionForm.py`
- `Feature\GUI\FeatureMergeForm.py`
- `Feature\GUI\IccEstimationForm.py`
- `Feature\GUI\RadiomicsParams.yaml`

其中 `Feature\GUI\FeatureExtractionForm.py` 是 radiomics GUI 工作流的重要入口，内部会使用：

- `UniqueFileMatcherManager`
- `SeriesStringMatcher`
- `radiomics.featureextractor.RadiomicsFeatureExtractor`
- `SimpleITK`

### `BC\`

二分类建模模块，是当前仓库中结构最完整、最值得优先理解的一块。

主要子目录职责：

- `BC\DataContainer\`：特征矩阵、标签、病例 ID 等数据容器
- `BC\FeatureAnalysis\`：数据平衡、归一化、降维、特征选择、分类器、交叉验证与 pipeline
- `BC\GUI\`：二分类相关 GUI 页面
- `BC\Visualization\`：ROC 等可视化
- `BC\Description\`：PDF 报告生成
- `BC\HyperParameters\`：模型和选择器超参数配置
- `BC\HyperParamManager\`：超参数读取/管理
- `BC\Image2Feature\`：图像到特征的补充流程
- `BC\Utility\`：辅助工具
- `BC\Func\`：指标和工具函数

关键文件：

- `BC\DataContainer\DataContainer.py`
- `BC\FeatureAnalysis\Pipelines.py`
- `BC\FeatureAnalysis\Classifier.py`
- `BC\FeatureAnalysis\FeatureSelector.py`
- `BC\FeatureAnalysis\DataBalance.py`

可把 BC 主流程理解为：

`读取结构化特征数据 -> 预处理/平衡 -> 降维/筛选 -> 分类器训练 -> 交叉验证 -> 可视化/报告/预测`

### `SA\`

生存分析模块，整体思路与 BC 平行，但围绕 survival task 构建。

关键文件：

- `SA\PipelineManager.py`
- `SA\DataContainer.py`
- `SA\Normalizer.py`
- `SA\DimensionReducer.py`
- `SA\FeatureSelector.py`
- `SA\Fitter.py`
- `SA\CrossValidation.py`
- `SA\GUI\ProcessForm.py`
- `SA\GUI\VisualizationForm.py`

可以简单记忆为：

**SA 是 BC 流程在生存分析场景下的平行版本。**

### `Plugin\`

插件发现与启动模块。

关键文件：

- `Plugin\PluginManager.py`

当前机制不是“进程内插件 API”，而更像“外部工具启动器”：

- 扫描 `Plugin\` 下的子目录
- 每个插件目录要求提供 `config.json`
- `config.json` 至少需要 `name` 与 `path`
- 可选提供 `format`、`figure`、`description`
- 首页读取插件图标与说明
- 执行时通过 `os.system(...)` 启动目标路径

当前仓库里暂未看到实际插件目录内容，说明插件框架存在，但仓库本身并未附带完整插件实例。

### `Utility\`

全局公共工具目录。

当前在首页路径中能直接确认使用的是：

- `Utility\EcLog.py`

### `src\`

注意：这里 **不是源码目录**。

`src\` 当前主要放的是图片资源，例如：

- `about.png`
- `Prepare.png`
- `Process.png`
- `Report.png`
- `Visualization.png`

后续开发不要被目录名误导。

### 其他目录/文件

- `Example\`：示例数据
- `log\`：日志目录
- `FECA.log`：运行日志
- `AI_Relate\`：项目分析/记忆文档
- `BuildScript.bat`：当前打包脚本
- `Release.bat`：旧发布脚本，已明显滞后
- `install.bat`：依赖安装脚本
- `MainFrameCall.spec`：当前 PyInstaller 规格文件

---

## 4. 三大业务域怎么记

如果只想快速回忆项目职责，可以记成：

- `Feature` = 从医学影像与 ROI 出发生成特征表
- `BC` = 用结构化特征做二分类模型开发与分析
- `SA` = 用结构化特征做生存分析模型开发与可视化

也就是：

**Feature 负责产出特征，BC/SA 负责消费特征并建模。**

---

## 5. 关键启动与模块关系

### 5.1 实际启动顺序

推荐记住下面这条链路：

1. 运行 `MainFrameCall.py` 或 `MainFrameCall_opt.py`
2. 创建 `QApplication`
3. 实例化 `HomePageForm`
4. 首页显示版本号并绑定各个按钮
5. 首页创建并持有 Feature / BC / SA / Plugin 对应的子窗体或管理器
6. 用户通过首页按钮进入具体模块

### 5.2 首页持有的对象

`HomeUI\HomePageForm.py` 中可直接看到以下对象被初始化：

- `FeatureExtractionForm`
- `FeatureMergeForm`
- `IccEstimationForm`
- `PrepareConnection`
- `ProcessConnection`
- `VisualizationConnection`
- `ModelPredictionForm`
- `SA.GUI.ProcessForm`
- `SA.GUI.VisualizationForm`
- `PluginManager`

这说明：

- 首页不仅负责导航，也负责跨模块实例生命周期的初始持有
- 各模块返回首页依赖 `close_signal`
- 后续若想重构导航，`HomePageForm.py` 是第一优先入口

---

## 6. 技术栈与依赖认知

### 6.1 当前可明确看到的核心依赖

- GUI：`PySide6`
- 数值与表格：`numpy`、`pandas`
- 机器学习：`scikit-learn`
- 类别不平衡：`imbalanced-learn`
- 放射组学：`pyradiomics`
- 医学图像：`SimpleITK`
- 生存分析：`lifelines`
- 绘图：`matplotlib`、`seaborn`
- 报告输出：`reportlab`、`pdfdocument`
- 打包：`PyInstaller`
- 统计：`scipy`、`statsmodels`、`pingouin`

### 6.2 依赖信息有历史漂移

当前仓库中依赖信息存在不完全一致的情况：

- `README.md` 中列出的依赖较新，且包含 `scikit-survival`、`trimesh`、`yaml`
- `install.bat` 中是较老的一组固定版本
- `MainFrameCall.spec` 又体现出实际打包时更关注 `scipy`、`pandas`、`matplotlib`、`imblearn`、`radiomics` 等

因此后续如果要真正跑通环境，不要只相信单一文件，建议交叉核对：

- `README.md`
- `install.bat`
- 实际代码 import
- `MainFrameCall.spec`

---

## 7. 打包、发布与版本管理现状

### 7.1 当前有效打包脚本

`BuildScript.bat` 是当前更可信的构建入口。

它会：

- 从 `HomeUI\VersionConstant.py` 读取版本号
- 调用 `pyinstaller`
- 使用 `MainFrameCall.spec`
- 输出到 `C:\Users\SunsServer\Project\FAE\FAEv<version>`

### 7.2 当前 spec 的实际含义

`MainFrameCall.spec` 不是单入口 spec，它会同时构建两个可执行集合：

- `fae`，对应 `MainFrameCall.py`
- `fae_opt`，对应 `MainFrameCall_opt.py`

它还会显式打包：

- `Feature\GUI\RadiomicsParams.yaml`
- `BC\HyperParameters\`
- `scipy` / `pandas` / `matplotlib` / `fontTools` / `imblearn` 数据
- `radiomics` 目录树

并排除：

- `torch`
- `IPython`
- `tkinter`

### 7.2.1 Qt 迁移备注

项目已从 `PyQt5` 迁移到 `PySide6`。后续如果继续维护界面层，需要注意：

- 手写窗体逻辑统一使用 `PySide6`
- 仓库内已生成的 UI Python 文件也必须保持 `from PySide6 import QtCore, QtGui, QtWidgets`
- matplotlib Qt 适配层当前依赖 `matplotlib.backends.backend_qtagg`

### 7.3 旧发布脚本已过时

`Release.bat` 目前明显滞后，原因包括：

- 里面仍写死 `FAEv0.6.6`
- 引用了 `MainFrameCall_optimized.spec`

但当前仓库中实际存在的是：

- `MainFrameCall.spec`
- `MainFrameCall_opt.py`

没有看到 `MainFrameCall_optimized.spec`。

所以：

**后续涉及发布时，应优先信任 `BuildScript.bat + MainFrameCall.spec`，不要直接依赖 `Release.bat`。**

---

## 8. 目前最值得优先理解的核心文件

如果未来要继续开发，建议优先读下面这些文件：

1. `README.md`
   - 获取产品定位、模块分区、依赖概览

2. `MainFrameCall.py`
   - 明确程序入口与运行目录处理

3. `HomeUI\HomePageForm.py`
   - 看懂模块地图、导航方式和首页初始化逻辑

4. `Feature\GUI\FeatureExtractionForm.py`
   - 看懂 radiomics 特征提取 GUI 的主工作流

5. `Feature\FileMatcher.py`
   - 理解病例、图像和 ROI 的匹配逻辑

6. `BC\FeatureAnalysis\Pipelines.py`
   - 理解二分类的总 pipeline 编排方式

7. `BC\DataContainer\DataContainer.py`
   - 明确特征数据结构和 CSV 输入输出方式

8. `SA\PipelineManager.py`
   - 理解生存分析的总 pipeline

9. `Plugin\PluginManager.py`
   - 理解插件发现机制和元信息格式

10. `BuildScript.bat` 与 `MainFrameCall.spec`
   - 理解当前构建与打包方式

---

## 9. 当前代码结构的典型特点

### 9.1 UI 与业务逻辑耦合偏重

大量功能模块遵循如下组合：

- `.ui`：Qt Designer 原始文件
- `xxx.py`：`pyuic` 生成的界面代码
- `xxxForm.py` 或连接类：业务逻辑和信号槽处理

这类结构对 GUI 快速交付友好，但维护上要注意：

- 控件逻辑和算法流程容易混在一起
- 不利于单元测试
- 后续想做 CLI 化或服务化会较难

### 9.2 仓库按业务域平铺，而非按可安装包分层

当前仓库根目录直接平铺：

- `Feature`
- `BC`
- `SA`
- `HomeUI`
- `Plugin`

这在桌面项目中并不少见，但对新人来说，项目边界和复用层次不够明显。

### 9.3 配置来源分散

几个重要的配置来源包括：

- `Feature\GUI\RadiomicsParams.yaml`
- `BC\HyperParameters\**\*.json`
- 各模块 `.ui`
- `MainFrameCall.spec`
- `HomeUI\VersionConstant.py`

如果未来改行为或做重构，建议先建立“配置影响清单”。

---

## 10. 目前已确认的风险和遗留问题

### 10.1 项目文档与脚本存在版本漂移

当前至少有这些不一致：

- 版本常量已经是 `0.7.0`
- `Release.bat` 仍然写 `0.6.6`
- README 的依赖版本与 `install.bat` 不完全一致

### 10.2 UI 编译链依赖手工维护

`README.md` 明确说明 `.ui` 文件需要手工转成 `.py`。

这意味着修改 UI 后，必须注意同步更新对应生成文件，否则运行行为和界面源码可能不一致。

### 10.3 当前仓库中未发现自动化测试

按现有仓库搜索，没有看到明显的：

- `pytest`
- `unittest`
- `test_*.py`

所以任何修改都应优先采用：

- 小步改动
- 精准阅读上下文
- 必要时做手工运行验证

### 10.4 插件执行是直接外部调用

插件通过 `os.system(...)` 执行，意味着：

- 插件失败时错误处理能力有限
- 安全边界较弱
- 与主程序的交互协议也比较松散

如果后续要强化插件系统，需要重新定义运行接口。

---

## 11. 后续开发时的建议理解方式

### 11.1 把系统拆成四层来记

虽然代码没有显式这样分层，但理解时可以这么映射：

- 入口层：`MainFrameCall*.py`
- 导航层：`HomeUI\HomePageForm.py`
- 业务层：`Feature` / `BC` / `SA`
- 扩展层：`Plugin`

### 11.2 把 Feature / BC / SA 的关系记成上下游

- `Feature` 偏数据准备和特征生成
- `BC` / `SA` 偏模型训练、验证、可视化和结果输出

以后排查问题时，先判断问题落在哪条链路：

- 图像与 ROI 匹配问题 -> `Feature`
- 特征表与标签载入问题 -> `BC` 或 `SA` 的 `DataContainer`
- 模型搜索、筛选、训练问题 -> `Pipelines.py` 或 `PipelineManager.py`
- 页面跳转和模块连接问题 -> `HomePageForm.py`

---

## 12. 给后续会话的快速热启动模板

如果下一次进入仓库，只想快速恢复上下文，建议按以下顺序做：

### 最短阅读顺序

1. `AI_Relate\FAE_Project_Memory.md`
2. `README.md`
3. `MainFrameCall.py`
4. `HomeUI\HomePageForm.py`
5. 与当前任务相关的业务模块

### 如果要继续做架构或功能开发

优先再读：

1. `Feature\GUI\FeatureExtractionForm.py`
2. `BC\FeatureAnalysis\Pipelines.py`
3. `SA\PipelineManager.py`
4. `Plugin\PluginManager.py`
5. `BuildScript.bat`
6. `MainFrameCall.spec`

---

## 13. 一句话总记忆

**FAE 是一个由 `MainFrameCall.py` / `MainFrameCall_opt.py` 启动、由 `HomeUI\HomePageForm.py` 统一导航的 PyQt 桌面放射组学工具；其核心由 `Feature`、`BC`、`SA` 三大模块组成，使用 PyInstaller 打包，并带有一个基于 `config.json` + `os.system(...)` 的轻量插件机制。**

