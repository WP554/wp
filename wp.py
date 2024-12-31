import streamlit as st
import requests
import re
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import seaborn as sns
import validators
import altair as alt
import plotly.express as px
import pygal
from pygal.style import LightColorizedStyle as LCS

# Set the font path, ensure the path points to a valid Chinese font
font_path = 'Font/SimHei.ttf'  # Please modify according to the actual font path
font_prop = fm.FontProperties(fname=font_path)


# URL fetch function
def fetch_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check if the request was successful
    response.encoding = 'utf-8'
    return response.text


def remove_html_tags(html):
    """Remove HTML tags"""
    return re.sub(r'<[^>]+>', '', html)


def remove_punctuation_and_english(text):
    """Remove punctuation and English characters, keep Chinese characters"""
    return re.sub(r'[A-Za-z0-9\s+]|[^\u4e00-\u9fa5]+', '', text)


# Generate word cloud
def generate_wordcloud(word_counts):
    wordcloud = WordCloud(font_path=font_path, width=400, height=200,
                          background_color='white').generate_from_frequencies(word_counts)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Do not show the axes
    fig = plt.gcf()
    return fig


# Visualization functions for Matplotlib
def plot_matplotlib(freq_df):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Bar chart
    axs[0, 0].bar(freq_df['词语'], freq_df['频率'], color='orange')
    axs[0, 0].set_title('柱状图')
    axs[0, 0].set_xlabel('词语')
    axs[0, 0].set_ylabel('频率')
    axs[0, 0].tick_params(axis='x', rotation=45)

    # Line chart
    axs[0, 1].plot(freq_df['词语'], freq_df['频率'], marker='o', color='b')
    axs[0, 1].set_title('折线图')
    axs[0, 1].set_xlabel('词语')
    axs[0, 1].set_ylabel('频率')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Pie chart
    axs[1, 0].pie(freq_df['频率'], labels=freq_df['词语'], autopct='%1.1f%%', startangle=90)
    axs[1, 0].set_title('饼图')
    axs[1, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Area chart
    axs[1, 1].fill_between(freq_df['词语'], freq_df['频率'], color='skyblue', alpha=0.5)
    axs[1, 1].set_title('面积图')
    axs[1, 1].set_xlabel('词语')
    axs[1, 1].set_ylabel('频率')
    axs[1, 1].tick_params(axis='x', rotation=45)

    # Scatter plot
    axs[2, 0].scatter(freq_df['词语'], freq_df['频率'], color='red')
    axs[2, 0].set_title('散点图')
    axs[2, 0].set_xlabel('词语')
    axs[2, 0].set_ylabel('频率')
    axs[2, 0].tick_params(axis='x', rotation=45)

    # Box plot
    sns.boxplot(x=freq_df['频率'], ax=axs[2, 1])
    axs[2, 1].set_title('箱线图')
    axs[2, 1].set_xlabel('频率')

    plt.tight_layout()
    return fig


# New function for Plotly visualization
def plotly_visualizations(freq_df):
    plots = {}

    # Bar chart
    plots['柱状图'] = px.bar(freq_df, x='词语', y='频率', title='Plotly柱状图', text='频率', template='plotly_white')

    # Line chart
    plots['折线图'] = px.line(freq_df, x='词语', y='频率', title='Plotly折线图', markers=True)

    # Pie chart
    plots['饼图'] = px.pie(freq_df, values='频率', names='词语', title='Plotly饼图')

    # Area chart
    plots['面积图'] = px.area(freq_df, x='词语', y='频率', title='Plotly面积图')

    # Scatter plot
    plots['散点图'] = px.scatter(freq_df, x='词语', y='频率', title='Plotly散点图')

    # Box plot
    plots['箱线图'] = px.box(freq_df, y='频率', title='Plotly箱线图')

    return plots


# Function for Altair visualizations
def altair_visualizations(freq_df):
    plots = {}

    # Bar chart
    plots['柱状图'] = alt.Chart(freq_df).mark_bar().encode(
        x='词语',
        y='频率',
        tooltip=['词语', '频率']
    ).properties(
        title='Altair柱状图'
    )

    # Line chart
    plots['折线图'] = alt.Chart(freq_df).mark_line(point=True).encode(
        x='词语',
        y='频率',
        tooltip=['词语', '频率']
    ).properties(
        title='Altair折线图'
    )

    # Pie chart - Alternative using a bar and transform
    plots['饼图'] = alt.Chart(freq_df).mark_arc().encode(
        theta='频率',
        color='词语'
    ).properties(
        title='Altair饼图'
    )

    # Area chart
    plots['面积图'] = alt.Chart(freq_df).mark_area(opacity=0.5).encode(
        x='词语',
        y='频率'
    ).properties(
        title='Altair面积图'
    )

    # Scatter plot
    plots['散点图'] = alt.Chart(freq_df).mark_circle(size=60).encode(
        x='词语',
        y='频率',
        tooltip=['词语', '频率']
    ).properties(
        title='Altair散点图'
    )

    # Box plot
    plots['箱线图'] = alt.Chart(freq_df).mark_boxplot().encode(
        y='频率'
    ).properties(
        title='Altair箱线图'
    )

    return plots


# Function for Pygal visualizations
def pygal_visualizations(freq_df):
    plots = {}

    bar_chart = pygal.Bar(style=LCS)
    bar_chart.title = 'Pygal柱状图'
    for index, row in freq_df.iterrows():
        bar_chart.add(row['词语'], row['频率'])
    plots['柱状图'] = bar_chart

    line_chart = pygal.Line(style=LCS)
    line_chart.title = 'Pygal折线图'
    line_chart.x_labels = freq_df['词语'].tolist()
    line_chart.add('频率', freq_df['频率'].tolist())
    plots['折线图'] = line_chart

    pie_chart = pygal.Pie(style=LCS)
    pie_chart.title = 'Pygal饼图'
    for index, row in freq_df.iterrows():
        pie_chart.add(row['词语'], row['频率'])
    plots['饼图'] = pie_chart

    area_chart = pygal.Line(style=LCS)
    area_chart.title = 'Pygal面积图'
    area_chart.x_labels = freq_df['词语'].tolist()
    area_chart.add('频率', freq_df['频率'].tolist(), stroke_style={'width': 2, 'color': 'blue'})
    plots['面积图'] = area_chart

    scatter_chart = pygal.XY(style=LCS)
    scatter_chart.title = 'Pygal散点图'
    scatter_chart.add('频率', [(i, freq) for i, freq in zip(freq_df['词语'], freq_df['频率'])])
    plots['散点图'] = scatter_chart

    box_chart = pygal.Box(style=LCS)
    box_chart.title = 'Pygal箱线图'
    box_chart.add('频率', freq_df['频率'].tolist())
    plots['箱线图'] = box_chart

    return plots


def main():
    st.title("文本分析与词频可视化")

    stopwords_file = st.sidebar.file_uploader("上传停用词文件 (stopwords.txt):", type=['txt'])
    url = st.sidebar.text_input("输入文章的 URL:", placeholder="https://example.com")

    # Create two dropdowns for library and chart selection
    visualization_type = st.sidebar.selectbox("选择可视化库:", ["Matplotlib", "Plotly", "Altair", "Pygal"])
    chart_type = st.sidebar.selectbox("选择图表类型:",
                                      ["柱状图", "折线图", "饼图", "面积图", "散点图", "箱线图"])

    if st.sidebar.button("抓取并分析"):
        if not validators.url(url):
            st.error("请输入有效的 URL!")
            return

        if stopwords_file is not None:
            stopwords = set(stopwords_file.read().decode('utf-8').splitlines())
        else:
            st.error("请上传停用词文件!")
            return

        try:
            html = fetch_content(url)
            clean_text = remove_html_tags(html)
            clean_text = remove_punctuation_and_english(clean_text)
            words = jieba.lcut(clean_text)

            meaningful_words = [word for word in words if word not in stopwords and len(word) > 1]
            word_counts = Counter(meaningful_words)
            most_common_words = word_counts.most_common(20)
            freq_df = pd.DataFrame(most_common_words, columns=['词语', '频率'])

            with st.expander("点击查看提取的文章"):
                st.write(clean_text)

            st.write("词频排名前 20 的词汇：")
            st.dataframe(freq_df)

            # Generate the corresponding visualizations based on user selection
            if chart_type == "词云图":
                st.write("生成词云图：")
                wordcloud_fig = generate_wordcloud(word_counts)
                st.pyplot(wordcloud_fig)  # Display word cloud
            else:
                if visualization_type == "Matplotlib":
                    fig = plot_matplotlib(freq_df)
                    st.pyplot(fig)

                elif visualization_type == "Plotly":
                    plots = plotly_visualizations(freq_df)
                    st.plotly_chart(plots[chart_type])

                elif visualization_type == "Altair":
                    plots = altair_visualizations(freq_df)
                    st.altair_chart(plots[chart_type], use_container_width=True)

                elif visualization_type == "Pygal":
                    plots = pygal_visualizations(freq_df)
                    st.write(plots[chart_type].render(is_unicode=True), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"发生错误：{e}")


if __name__ == "__main__":
    main()  