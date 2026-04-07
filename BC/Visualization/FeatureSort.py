import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

color = sns.color_palette('deep') + sns.color_palette('bright')


def FeatureSort(feature_name, group=np.array(()), group_name=[], value=[], store_path='',
                is_sort=True, is_show=True, fig=plt.figure()):
    '''
    Draw the plot of the sorted feature, an option is to draw different color according to the group and group value.

    :param feature_name: The name of the features
    :param group: the array to map the feature name to the group name, default is zeros
    :param group_name: the group name list to denote the group. default is []
    :param value: The value of each feature_name. Default is []
    :param store_path: The store path, supporting .jpeg and .tif
    :param is_sort: Boolen, to sort the features according to value. Default is True
    :return:

    Apr-29-18, Yang SONG [yang.song.91@foxmail.com]
    '''
    if group.size == 0 and group_name == []:
        group = np.zeros((len(feature_name), ), dtype=np.uint8)
        group_name = ['']

    if value == []:
        value = [len(feature_name) - index for index in range(len(feature_name))]
    else:
        value = np.abs(np.squeeze(value))

    if is_sort:
        sort_index = sorted(range(len(value)), key=lambda k: value[k], reverse=True)
        
        value = [value[index] for index in sort_index]
        feature_name = [feature_name[index] for index in sort_index]
        group = [group[index] for index in sort_index]

    assert(np.max(group) + 1 == len(group_name))
    sub_group = np.zeros((len(group_name), len(feature_name)))
    for index in range(len(feature_name)):
        sub_group[group[index], index] = value[index]
    y = range(len(feature_name))

    fig.clear()
    ax = fig.add_subplot(111)

    for index in range(sub_group.shape[0]):
        ax.barh(y, sub_group[index, :], color=color[index])

    ax.set_yticks(range(len(feature_name)))
    ax.set_yticklabels(feature_name)
    ax.set_xticks([])
    if len(group_name) > 1:
        ax.legend(group_name)

    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            fig.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        fig.show()

    return ax

def ShortFeatureFullName(feature_full_name):
    if len(feature_full_name) <= 5:
        return feature_full_name

    sub = re.findall("[A-Z]", feature_full_name)
    if len(sub) == 1:
        return feature_full_name[:5]
    elif len(sub) == 0:
        return feature_full_name[:5]
    else:
        return ''.join(sub)

def SeperateRadiomicsFeatures(feature_name):
    '''
    Generate the feature name, group, and group cound according to radiomics features.

    :param feature_name: The generated radiomis featues. which should including sequence_name, image_class,
    feature_class, and the feature_name, like T2_origin_glszm_ZonePercentage
    :return:

    Apr-29-18 Yang SONG [yang.song.91@foxmail.com]
    '''
    sub_feature_name = []
    group = []
    group_name = []

    seq_group = []
    seq_count = 0
    image_class_group = []
    image_class_count = 0
    feature_class_group = []
    feature_class_count = 0

    for feature in feature_name:
        sep = feature.split('_')
        if len(sep) == 2:
            sep = [sep[0], '', '', sep[1]]

        seq = sep[0]
        image_class = sep[1]
        feature_class = sep[-2]

        if not seq in seq_group:
            seq_count += 1
        seq_group.append(seq)
        if not image_class in image_class_group:
            image_class_count += 1
        image_class_group.append(image_class)
        if not feature_class in feature_class_group:
            feature_class_count += 1
        feature_class_group.append(feature_class)

        sub_feature_name.append(ShortFeatureFullName(sep[-1]))

    if seq_count == 1:
        seq_group = ['' for index in range(len(feature_name))]
    if image_class_count == 1:
        image_class_group = ['' for index in range(len(feature_name))]
    if feature_class_count == 1:
        feature_class_group = ['' for index in range(len(feature_name))]

    group_count = 0
    for index in range(len(feature_name)):
        temp_name = seq_group[index] + '-' + image_class_group[index] + '-' + feature_class_group[index]
        if not temp_name in group_name:
            group_name.append(temp_name)
            group.append(group_count)
            group_count += 1
        else:
            group.append(group_name.index(temp_name))

    return sub_feature_name, np.asarray(group, dtype=np.uint8), group_name

def SortRadiomicsFeature(feature_name, value=[], store_path='', is_show=False, fig=plt.figure()):
    sub_feature_name, group, group_name = SeperateRadiomicsFeatures(feature_name)
    FeatureSort(sub_feature_name, group, group_name, value, store_path, is_show=is_show, fig=fig)

def GeneralFeatureSort(feature_name, value=[], store_path='', is_sort=True, max_num=-1, is_show=True, fig=plt.figure(), reverse=True):
    if not isinstance(value, list):
        value = list(value)
    if value == []:
        value = [len(feature_name) - index for index in range(len(feature_name))]

    if is_sort:
        sort_index = sorted(range(len(value)), key=lambda k: value[k], reverse=reverse)

        value = [value[index] for index in sort_index]
        feature_name = [feature_name[index] for index in sort_index]

    if max_num > 0:
        value = value[:max_num]
        feature_name = feature_name[:max_num]

    fig.clear()
    # margin = 0.2

    left, bottom, width, height = 0.75, 0.1, 0.2, 0.8

    ax = fig.add_axes([left, bottom, width, height])
    # ax = fig.add_subplot(111)
    ax.barh(range(len(feature_name)), value, color=color[0])
    ax.set_yticks(range(len(feature_name)))
    ax.set_yticklabels(feature_name)

    ax.set_xticks([])



    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'jpg':
            plt.savefig(store_path, dpi=300, format='jpeg')
        elif store_path[-3:] == 'eps':
            plt.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        fig.show()

    return ax

if __name__ == '__main__':
    # feature_name = ['DE', 'SAE', 'SZNUM', 'JE', 'Id']
    feature_name = ['10Per', 'Autoc', 'IR', 'GLV']
    value = [72.9, 45.4, 45.2, 41]
    group = np.array([1, 3, 2, 0])
    group_name = ['ADC--firstorder', 'DWI--firstorder', 'DWI--glcm', 'DWI--glszm']
    # group = [0, 1, 1, 0, 0]
    # group_name = ['GLCM', 'GLSZM']
    # value = 0.1, 0.5, 0.9, 0.2, 0.1
    FeatureSort(feature_name, group, group_name, value, is_show=True)

    # import pandas as pd
    # df = pd.read_csv(r'C:\Users\yangs\Desktop\anova_sort.csv', index_col=0)
    # feature_name = list(df.index)
    # value = list(df['F'])
    # new_feature_name = [ShortFeatureFullName(index) for index in feature_name]
    # GeneralFeatureSort(new_feature_name, value, max_num=4, is_show=True, store_path=r'D:\MyDocs\Document\研究生\毕业\毕业论文\图\组学\ANOVA_sort.jpg')
    # SortRadiomicsFeature(new_feature_name, value, is_show=True)

def SHAPBarPlot(shap_df, max_num=20, is_show=False, fig=None):
    """Horizontal bar chart sorted by signed mean SHAP with RdBu_r gradient coloring.

    Features are ordered positive-max (top) -> negative-min (bottom).
    Bar length = |mean SHAP|; color encodes sign and magnitude via RdBu_r colormap.

    Args:
        shap_df: pd.DataFrame (n_samples, n_features), signed SHAP values.
        max_num: Top N features to display (selected by |mean SHAP|, then sorted by sign).
        is_show: Whether to call fig.show().
        fig: matplotlib Figure object.

    Returns:
        matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if fig is None:
        fig = plt.figure()

    mean_shap = shap_df.mean(axis=0)   # signed mean per feature

    # Step 1: Select top-N by |mean SHAP|
    if max_num > 0:
        top_idx = mean_shap.abs().nlargest(max_num).index
    else:
        top_idx = mean_shap.index

    # Step 2: Sort by |mean SHAP| descending -> most important at index 0 (bottom of barh)
    abs_sorted_idx = mean_shap[top_idx].abs().sort_values(ascending=False).index
    sorted_series = mean_shap[top_idx].reindex(abs_sorted_idx)
    labels = list(sorted_series.index)
    signed_vals = sorted_series.values          # signed, for coloring
    bar_lengths = np.abs(signed_vals)           # always positive, for bar width

    # Step 3: Map signed values -> RdBu_r colors
    v_max = np.abs(signed_vals).max() if len(signed_vals) > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max)
    cmap = cm.get_cmap('RdBu_r')
    colors = [cmap(norm(v)) for v in signed_vals]

    # Step 4: Draw
    fig.clear()
    ax = fig.add_axes([0.42, 0.08, 0.42, 0.84])

    ax.barh(range(len(labels)), bar_lengths, color=colors, height=0.65,
            edgecolor='none')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=9, color='#555555')
    ax.set_title('Feature Contribution (SHAP)', fontsize=10, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    # Step 5: Colorbar — leave enough right margin for tick labels
    cax = fig.add_axes([0.86, 0.08, 0.03, 0.84])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('mean SHAP', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if is_show:
        fig.show()
    return ax

def SHAPBeeswarmPlot(shap_df, feature_df=None, max_num=20, is_show=False, fig=None):
    """Beeswarm plot: each dot = one training sample, colored by feature value.

    Args:
        shap_df: pd.DataFrame (n_samples, n_features), signed SHAP values.
        feature_df: pd.DataFrame (n_samples, n_features), original feature values
                    used for dot coloring. If None, dots are gray.
        max_num: Number of top features to show (by mean |SHAP|).
        is_show: Whether to call fig.show().
        fig: matplotlib Figure object.

    Returns:
        matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colorbar import ColorbarBase

    if fig is None:
        fig = plt.figure()

    mean_abs = shap_df.abs().mean(axis=0)
    if max_num > 0:
        top_idx = mean_abs.nlargest(max_num).index
    else:
        top_idx = mean_abs.index
    # Sort by |mean SHAP| descending -> most important at index 0 (bottom of barh)
    sorted_idx = mean_abs[top_idx].sort_values(ascending=False).index

    fig.clear()
    ax = fig.add_axes([0.42, 0.08, 0.48, 0.84])

    cmap = cm.get_cmap('RdBu_r')
    rng = np.random.default_rng(42)

    for y_pos, feat in enumerate(sorted_idx):
        shap_vals = shap_df[feat].values
        n = len(shap_vals)

        if feature_df is not None and feat in feature_df.columns:
            feat_vals = feature_df[feat].values.astype(float)
            feat_min, feat_max = feat_vals.min(), feat_vals.max()
            norm_vals = (feat_vals - feat_min) / (feat_max - feat_min + 1e-8)
            dot_colors = cmap(norm_vals)
        else:
            dot_colors = np.full((n, 4), [0.5, 0.5, 0.5, 0.7])

        jitter = rng.uniform(-0.25, 0.25, size=n)
        ax.scatter(shap_vals, y_pos + jitter, c=dot_colors,
                   s=12, alpha=0.7, linewidths=0)

    ax.axvline(0, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(list(sorted_idx), fontsize=9)
    ax.set_xlabel('SHAP value', fontsize=9, color='#555555')
    ax.set_title('Feature Contribution (SHAP Beeswarm)', fontsize=10, pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

    # Colorbar for feature value
    if feature_df is not None:
        cax = fig.add_axes([0.91, 0.08, 0.02, 0.84])
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('Feature value', fontsize=8)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Low', 'High'])

    if is_show:
        fig.show()
    return ax



