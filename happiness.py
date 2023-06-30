# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, calinski_harabasz_score
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------- RUN THE APP ----------------------
# cd C:\Users\Edoardo\Documents\GitHub\Social_Research_2022_2023-HappinessCorruption
# streamlit run happiness.py

# ---------------------- IMPORT DATASET ----------------------
happiness_df = pd.read_csv('WorldHappiness_Corruption_2015_2020.csv')

st.title('Does money make happiness?')
st.subheader('Yes, but not only.')
st.write('''
    The World Happiness Report is a landmark survey of the state of global happiness that ranks 791 countries by 
    how happy their citizens perceive themselves to be.
    The data used in this dataset Sustainable Development Solutions Network's and the data is referred to 2015-2020.
    ''')

# ---------------------- COVER IMAGE ----------------------
image = Image.open('cover.jfif')
st.image(image)
sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(10,6))

# ---------------------- SIDEBAR ----------------------
st.sidebar.subheader('Sections')

# ---------------------- DATASET SECTION ----------------------
if st.sidebar.checkbox('Dataset'):
    st.header('Happiness Dataset')
    st.write(happiness_df)
    st.write('''
        - **happiness_score**: average of responses to the primary life evaluation question from the Gallup World Poll (GWP).
        - **gdp_per_capita**: the extent to which GDP contributes to the calculation of the Happiness Score.
        - **family**: the extent to which Family contributes to the calculation of the Happiness Score.
        - **health**: the extent to which Life expectancy contributed to the calculation of the Happiness Score.
        - **freedom**: the extent to which Freedom contributed to the calculation of the Happiness Score.
        - **generosity**: a numerical value calculated based on poll participants' perceptions of generosity in their country.
        - **government_trust**: the extent to which Perception of Corruption contributes to Happiness Score.
        - **dystopia_residual**: a score based on a hypothetical comparison to the world's saddest country.
        - **continent**: region of the country.
        - **year**: date of each of the findings.
        - **social_support**: the perception and actuality that one is cared for, has assistance available from other people,
          and most popularly, that one is part of a supportive social network.
        - **cpi_score**: corruption perception index (the higher the better).
        ''')

# ---------------------- CORRELATION MATRIX SECTION ----------------------
if st.sidebar.checkbox('Correlation Matrix'):
    st.header('Correlation Matrix')
    corr = happiness_df.corr()
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax1)
    st.pyplot()
    
    st.markdown('''
        The Correlation Matrix show that the :green[**most correlated variables**] with Happiness are:
        - **GDP per Capita** :green[(0.79)]
        - **Health Score** :green[(0.75)]
        - **CPI Score** :green[(0.69)]
        - **Freedom Score** :green[(0.54)]
    ''')
    st.markdown('''
        The :red[**least correlated variables**] with Happiness score are:
        - **Family Score** :red[(0.15)]
        - **Generosity Score** :red[(0.16)]
        - **Dystopia Residual** :red[(0.17)]
        - **Social Support** :red[(0.19)]
        ''')

# ---------------------- DESCRIPTIVE STATISTICS & DISTRIBUTION SECTION ----------------------
if st.sidebar.checkbox('Descriptive Statistics & Distribution'):
    st.header('Descriptive Statistics & Distribution')
    st.subheader('Descriptive Statistics')
    st.write(happiness_df.describe())
    st.write("---")

    st.subheader('Distribution in histograms')
    # grid of distribution plots for each variable with titles
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    sns.histplot(happiness_df['happiness_score'], kde=True, ax=ax[0, 0], color='#a6cee3').set_title('Distribution of Happiness Score')
    sns.histplot(happiness_df['gdp_per_capita'], kde=True, ax=ax[0, 1], color='#1f78b4').set_title('Distribution of GDP per Capita')
    sns.histplot(happiness_df['cpi_score'], kde=True, ax=ax[0, 2], color='#b2df8a').set_title('Distribution of CPI Score')
    sns.histplot(happiness_df['health'], kde=True, ax=ax[1, 0], color='#33a02c').set_title('Distribution of Health Score')
    sns.histplot(happiness_df['freedom'], kde=True, ax=ax[1, 1], color='#fb9a99').set_title('Distribution of Freedom Score')
    sns.histplot(happiness_df['social_support'], kde=True, ax=ax[1, 2], color='#e31a1c').set_title('Distribution of Social Support')
    sns.histplot(happiness_df['generosity'], kde=True, ax=ax[2, 0], color='#fdbf6f').set_title('Distribution of Generosity Score')
    sns.histplot(happiness_df['family'], kde=True, ax=ax[2, 1], color='#ff7f00').set_title('Distribution of Family Score')
    sns.histplot(happiness_df['dystopia_residual'], kde=True, ax=ax[2, 2], color='#cab2d6').set_title('Distribution of Dystopia Residual')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between subplots
    st.pyplot()

    st.markdown('''
        - **Happiness Score**:
            - Mean value ($\mu$): 5.5
            - Standard Deviation ($\sigma$): 1.1
        - **GDP per Capita**:
            - $\mu$: 9.2
            - $\sigma$: 1.2
        - **CPI Score**:
            - $\mu$: 42.5
            - $\sigma$: 19.5
        - **Health Score**:
            - $\mu$: 63.5
            - $\sigma$: 7.2
        - **Freedom Score**:
            - $\mu$: 78.0
            - $\sigma$: 12.5
        - **Social Support**:
            - $\mu$: 81.3
            - $\sigma$: 12.1
        - **Generosity Score**:
            - $\mu$: 45.5
            - $\sigma$: 9.9
        - **Family Score**:
            - $\mu$: 81.2
            - $\sigma$: 12.1
        - **Dystopia Residual**:
            - $\mu$: 2.3
            - $\sigma$: 0.5
        ''')        
    
    # boxplots for each variable with titles
    st.write("---")
    st.subheader('Distribution in boxplots')
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    sns.boxplot(x=happiness_df['happiness_score'], ax=ax[0, 0], color='#a6cee3').set_title('Boxplot of Happiness Score')
    sns.boxplot(x=happiness_df['gdp_per_capita'], ax=ax[0, 1], color='#1f78b4').set_title('Boxplot of GDP per Capita')
    sns.boxplot(x=happiness_df['cpi_score'], ax=ax[0, 2], color='#b2df8a').set_title('Boxplot of CPI Score')
    sns.boxplot(x=happiness_df['health'], ax=ax[1, 0], color='#33a02c').set_title('Boxplot of Health Score')
    sns.boxplot(x=happiness_df['freedom'], ax=ax[1, 1], color='#fb9a99').set_title('Boxplot of Freedom Score')
    sns.boxplot(x=happiness_df['social_support'], ax=ax[1, 2], color='#e31a1c').set_title('Boxplot of Social Support')
    sns.boxplot(x=happiness_df['generosity'], ax=ax[2, 0], color='#fdbf6f').set_title('Boxplot of Generosity Score')
    sns.boxplot(x=happiness_df['family'], ax=ax[2, 1], color='#ff7f00').set_title('Boxplot of Family Score')
    sns.boxplot(x=happiness_df['dystopia_residual'], ax=ax[2, 2], color='#cab2d6').set_title('Boxplot of Dystopia Residual')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the spacing between subplots
    st.pyplot()
    st.write('''
        - **Happiness Score**:
            - _Median value_: 5.4
            - _Range_: [3.4-7.8]
        - **GDP per Capita**:
            - _Median value_: 9.4
            - _Range_: [6.5-11.5]
        - **CPI Score**:
            - _Median value_: 43.0
            - _Range_: [9.0-90.0]
        - **Health Score**:
            - _Median value_: 64.0
            - _Range_: [32.0-95.0]
        - **Freedom Score**:
            - _Median value_: 82.0
            - _Range_: [1.0-100]
        - **Social Support**:
            - _Median value_: 83.0
            - _Range_: [0.0-100.0]
        - **Generosity Score**:
            - _Median value_: 46.0
            - _Range_: [0.0-100.0]
        - **Family Score**:
            - _Median value_: 83.0
            - _Range_: [0.0-100.0]
        - **Dystopia Residual**:
            - _Median value_: 2.3
            - _Range_: [0.0-3.8]
        ''')

# ------------------------------ HAPPINESS SCORE TRENDS ------------------------------
if st.sidebar.checkbox('Happiness Score Trends'):
    
    # ------------------------------ Happiness by Continent in the years ------------------------------
    st.header('Happiness by Continent in the years')
    st.subheader('Happiness vs GDP per Capita')
    # ------------------------------ Happiness Score and GDP per Capita in all Continents ------------------------------
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('GDP per capita')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs GDP per capita in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('GDP per capita')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs GDP per capita in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('GDP per capita')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs GDP per capita in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('GDP per capita')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs GDP per capita in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('GDP per capita')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs GDP per capita in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['gdp_per_capita', 'happiness_score']]
        # Plotting GDP per capita on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['gdp_per_capita'], label='GDP per capita', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('GDP per capita')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs GDP per capita in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot(fig)

        st.caption('''
            The above plots show the relationship between Happiness and GDP per capita in each continent from 2015 to 2020.
            We can see that in the most of the cases there is a positive correlation between Happiness and GDP per Capita in each continent.
            ''')
    
    # ........... Happiness vs Health ...........
    st.subheader('Happiness vs Health')
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Health Score')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs Health in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('Health Score')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs Health in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Health Score')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs Health in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Health Score')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs Health in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('Health Score')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs Health in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['health', 'happiness_score']]
        # Plotting Health Score on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['health'], label='Health Score', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('Health Score')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs Health in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot(fig)

        st.caption('''
            The above plots show the relationship between Happiness and Health in each continent from 2015 to 2020.
            We can see that in the most of the cases there is a positive correlation between Happiness and Health except for
            Asia and Australia.
            ''')

    # ................ Happiness vs Corruption ................
    st.subheader('Happiness vs Corruption')
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Corruption perception')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs Corruption in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('Corruption perception')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs Corruption in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Corruption perception')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs Corruption in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Corruption perception')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs Corruption in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('Corruption perception')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs Corruption in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['cpi_score', 'happiness_score']]
        # Plotting Corruption perception on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['cpi_score'], label='Corruption perception', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('Corruption perception')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs Corruption in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot(fig)

        st.caption('''
            The above plots show the relationship between Happiness and Corruption in each continent from 2015 to 2020.
            We can see that in the most of the cases there is a positive correlation between Happiness and Corruption.
            ''')

    # ............... Happiness vs Family ...............
    st.subheader('Happiness vs Family')
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Family Score')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs Family in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('Family Score')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs Family in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Family Score')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs Family in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Family Score')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs Family in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('Family Score')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs Family in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['family', 'happiness_score']]
        # Plotting Family Score on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['family'], label='Family Score', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('Family Score')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs Family in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot(fig)

        st.caption('''
            The above plots show the relationship between Happiness and Family Score in each continent from 2015 to 2020.
            We can see that in Asia, in North America, in South America and in Australia there is a positive correlation 
            between Happiness and Family Score.
            ''')
    
    # ............ Happiness vs Generosity ............
    st.subheader('Happiness vs Generosity')
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Generosity Score')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs Generosity in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('Generosity Score')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs Generosity in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Generosity Score')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs Generosity in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Generosity Score')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs Generosity in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('Generosity Score')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs Generosity in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['generosity', 'happiness_score']]
        # Plotting Generosity Score on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['generosity'], label='Generosity Score', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('Generosity Score')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs Generosity in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot(fig)

        st.caption('''
            The above plots show the relationship between Happiness and Generosity Score in each continent from 2015 to 2020.
            We can see that in Europe and in Africa there is a negative correlation between Happiness and Generosity Score.
            ''')

    # ............ Happiness vs Social Support ............
    st.subheader('Happiness vs Social Support')
    with st.expander('Click here to show/hide the plots'):
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        # Europe
        df_Europe = happiness_df[happiness_df['continent'] == 'Europe']
        df_years_Europe = df_Europe[df_Europe['Year'].between(2015, 2020)]
        df_grouped = df_years_Europe.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[0, 0].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='#1A66CC')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('Social Support Score')
        axs[0, 0].tick_params(axis='y', labelcolor='#1A66CC')
        # Secondary y-axis for happiness score
        axs1 = axs[0, 0].twinx()
        axs1.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        axs1.set_ylabel('Happiness Score') 
        axs1.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 0].set_title('Happiness vs Social Support in Europe')
        # Displaying a single legend for both lines
        lines, labels = axs[0, 0].get_legend_handles_labels()
        lines2, labels2 = axs1.get_legend_handles_labels()
        axs1.legend(lines + lines2, labels + labels2, loc='best')

        # Africa
        df_Africa = happiness_df[happiness_df['continent'] == 'Africa']
        df_years_Africa = df_Africa[df_Africa['Year'].between(2015, 2020)]
        df_grouped = df_years_Africa.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[0, 1].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='green')
        axs[0, 1].set_xlabel('Year')
        axs[0, 1].set_ylabel('Social Support Score')
        axs[0, 1].tick_params(axis='y', labelcolor='green')
        # Creating a secondary y-axis for happiness score
        ax2 = axs[0, 1].twinx()
        ax2.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax2.set_ylabel('Happiness Score')
        ax2.tick_params(axis='y', labelcolor='#FFC30B')
        axs[0, 1].set_title('Happiness vs Social Support in Africa')
        # Displaying a single legend for both lines
        lines3, labels3 = axs[0, 1].get_legend_handles_labels()
        lines4, labels4 = ax2.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

        # Asia
        df_Asia = happiness_df[happiness_df['continent'] == 'Asia']
        df_years_Asia = df_Asia[df_Asia['Year'].between(2015, 2020)]
        df_grouped = df_years_Asia.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[1, 0].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='#FF4747')
        axs[1, 0].set_xlabel('Year')
        axs[1, 0].set_ylabel('Social Support Score')
        axs[1, 0].tick_params(axis='y', labelcolor='#FF4747')
        # Creating a secondary y-axis for happiness score
        ax3 = axs[1, 0].twinx()
        ax3.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax3.set_ylabel('Happiness Score')
        ax3.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 0].set_title('Happiness vs Social Support in Asia')
        # Displaying a single legend for both lines
        lines5, labels5 = axs[1, 0].get_legend_handles_labels()
        lines6, labels6 = ax3.get_legend_handles_labels()
        ax3.legend(lines5 + lines6, labels5 + labels6, loc='best')

        # North America
        df_North_America = happiness_df[happiness_df['continent'] == 'North America']
        df_years_North_America = df_North_America[df_North_America['Year'].between(2015, 2020)]
        df_grouped = df_years_North_America.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[1, 1].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='#FB7D10')
        axs[1, 1].set_xlabel('Year')
        axs[1, 1].set_ylabel('Social Support Score')
        axs[1, 1].tick_params(axis='y', labelcolor='#FB7D10')
        # Creating a secondary y-axis for happiness score
        ax4 = axs[1, 1].twinx()
        ax4.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax4.set_ylabel('Happiness Score')
        ax4.tick_params(axis='y', labelcolor='#FFC30B')
        axs[1, 1].set_title('Happiness vs Social Support in North America')
        # Displaying a single legend for both lines
        lines7, labels7 = axs[1, 1].get_legend_handles_labels()
        lines8, labels8 = ax4.get_legend_handles_labels()
        ax4.legend(lines7 + lines8, labels7 + labels8, loc='best')

        # South America
        df_South_America = happiness_df[happiness_df['continent'] == 'South America']
        df_years_South_America = df_South_America[df_South_America['Year'].between(2015, 2020)]
        df_grouped = df_years_South_America.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[2, 0].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='#813EB6')
        axs[2, 0].set_xlabel('Year')
        axs[2, 0].set_ylabel('Social Support Score')
        axs[2, 0].tick_params(axis='y', labelcolor='#813EB6')
        # Creating a secondary y-axis for happiness score
        ax5 = axs[2, 0].twinx()
        ax5.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax5.set_ylabel('Happiness Score')
        ax5.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 0].set_title('Happiness vs Social Support in South America')
        # Displaying a single legend for both lines
        lines9, labels9 = axs[2, 0].get_legend_handles_labels()
        lines10, labels10 = ax5.get_legend_handles_labels()
        ax5.legend(lines9 + lines10, labels9 + labels10, loc='best')

        # Australia
        df_Australia = happiness_df[happiness_df['continent'] == 'Australia']
        df_years_Australia = df_Australia[df_Australia['Year'].between(2015, 2020)]
        df_grouped = df_years_Australia.groupby('Year').mean()[['social_support', 'happiness_score']]
        # Plotting Social Support Score on the left y-axis
        axs[2, 1].plot(df_grouped.index, df_grouped['social_support'], label='Social Support Score', linestyle='-', color='#987554')
        axs[2, 1].set_xlabel('Year')
        axs[2, 1].set_ylabel('Social Support Score')
        axs[2, 1].tick_params(axis='y', labelcolor='#987554')
        # Creating a secondary y-axis for happiness score
        ax6 = axs[2, 1].twinx()
        ax6.plot(df_grouped.index, df_grouped['happiness_score'], label='Happiness Score', linestyle='-', color='#FFC30B')
        ax6.set_ylabel('Happiness Score')
        ax6.tick_params(axis='y', labelcolor='#FFC30B')
        axs[2, 1].set_title('Happiness vs Social Support in Australia')
        # Displaying a single legend for both lines
        lines11, labels11 = axs[2, 1].get_legend_handles_labels()
        lines12, labels12 = ax6.get_legend_handles_labels()
        ax6.legend(lines11 + lines12, labels11 + labels12, loc='best')

        plt.tight_layout()  # Ensures labels and titles are not cut off
        st.pyplot()

        st.caption('''
            The above plots show the relationship between Happiness and Social Support Score in each continent from 2015 to 2020.
            We can see that only in Europe and in Africa there is a positive correlation between Happiness and Social Support.
            ''')
    
# ---------------------------------- CLUSTERING SECTION ----------------------------------
if st.sidebar.checkbox('Clustering'):
    # ---------------------------------- K-Means++ Clustering ----------------------------------
    st.subheader('K-Means++ Clustering')

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # create a copy of the dataframe adding a new column based on the happiness score
    kmeans_df = happiness_df.copy()

    # remove the country with 0 on social support
    kmeans_df = kmeans_df[kmeans_df['social_support'] != 0]
    # Scale the variables
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    
    kmeans_df['cpi_score'] = scaler.fit_transform(kmeans_df[['cpi_score']])

    # Get available features for selection
    available_features = ['gdp_per_capita', 'health',
                          'freedom', 'family', 'generosity',
                          'government_trust', 
                          #'social_support', 
                          #'cpi_score', 
                          #'happiness_score'
                          ]
    
    # elbow method to find the optimal number of clusters
    X = kmeans_df[available_features].values
    X_scaled = scaler.fit_transform(X)
    
    # Find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method results
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, 11))
    n_clusters = 3  # Set a default value
    # Add a line indicating the best number of clusters
    plt.axvline(x=n_clusters, color='red', linestyle='--', label='Optimal Number of Clusters')
    plt.legend()
    st.pyplot()

    # Select features for clustering
    selected_features = st.multiselect('Select features for clustering', available_features)
    if selected_features:
        # Perform dimensionality reduction using PCA
        kmeans_df_selected = kmeans_df[selected_features]

        if len(selected_features) == 1:
            kmeans_df_pca = kmeans_df_selected.values.reshape(-1, 1)
        else:
            kmeans_df_pca = PCA(n_components=2).fit_transform(kmeans_df_selected)

        # Perform K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
        labels = kmeans.fit_predict(kmeans_df_pca)

        # Define country types and colors
        country_types = ['Unhappy Countries', 'Happy Countries', 'Neutral Countries']
        colors = ['red', 'green', 'blue']


        # Calculate the means of the selected clusters
        cluster_means = []
        for cluster in range(n_clusters):
            cluster_mean = kmeans_df.loc[labels == cluster, selected_features].mean()
            cluster_means.append(cluster_mean)

        # Plot the results
        plt.figure(figsize=(10, 8))
        if len(selected_features) == 1:
            plt.scatter(kmeans_df_pca[labels == country_types, 0], kmeans_df_pca[labels == country_types, 1], color=colors[country_types], label=country_types[country_types])
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')

        else:
            for i, label in enumerate(np.unique(labels)):
                plt.scatter(kmeans_df_pca[labels == label, 0], 
                            kmeans_df_pca[labels == label, 1], 
                            color=colors[label], 
                            label=country_types[label])

            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Centroids')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('K-Means Clustering Results')
        plt.legend()
        plt.show()
        st.pyplot()

        # Plot histograms with cluster means
        fig2, axes = plt.subplots(1, n_clusters, figsize=(12, 6))
        for cluster in range(n_clusters):
            axes[cluster].bar(selected_features, cluster_means[cluster])
            axes[cluster].set_title(f'Cluster {cluster+1}')
            axes[cluster].set_xlabel('Variables')
            axes[cluster].set_ylabel('Mean')
            axes[cluster].set_xticklabels(selected_features, rotation=45)
            axes[cluster].set_ylim([0, np.max(cluster_means) * 1.2])

        plt.tight_layout()
        st.pyplot(fig2)
        
        # Evaluate the clustering results
        st.subheader('Evaluation of the clustering results')
        st.write('Silhouette Score: ', silhouette_score(kmeans_df_pca, labels).round(2))
        st.write('A Silhouette Score of 0.5 or above indicates that the clustering results are good.')
        st.write('Calinski Harabasz Score: ', calinski_harabasz_score(kmeans_df_pca, labels).round(2))
        st.write('A Calinski Harabasz Score of 741 or above indicates that the clustering results are good.')
    else:
        st.write("Please select at least one feature for clustering.")

# ---------------------------------- CONCLUSION SECTION ----------------------------------
if st.sidebar.checkbox('Conclusion'):
    st.header('Conclusion')
    st.write('''
        In conclusion, we can see that the happiness score is not only influenced by the GDP per capita, 
        but also by other factors such as social support, freedom, generosity, and government trust.
        ''')