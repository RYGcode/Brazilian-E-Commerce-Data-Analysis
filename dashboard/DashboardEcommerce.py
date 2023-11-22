import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import plotly.express as px 

# load dataset
df = pd.read_csv("main_data.csv", compression='gzip')

# Dropping unused data columns
dropped_columns = ['order_id', 'product_description_lenght', 'product_photos_qty',
                   'product_length_cm', 'product_height_cm', 'product_width_cm']
df = df.drop(columns=dropped_columns)

# Changing the data type for date columns
timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                   'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in timestamp_cols:
    df[col] = pd.to_datetime(df[col])


# Create page configuration
st.set_page_config(page_title="Brazilian E-Commerce Data Analysis Dashboard",
                   page_icon="bar_chart:",
                   layout="wide")

# Create helper functions
def create_order_date_df(df):
    order_date_df = pd.DataFrame()

    # Extracting attributes for purchase date - Year and Month
    order_date_df['order_purchase_year'] = df['order_purchase_timestamp'].apply(lambda x: x.year)
    order_date_df['order_purchase_month'] = df['order_purchase_timestamp'].apply(lambda x: x.month)
    order_date_df['order_purchase_month_name'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))
    order_date_df['order_purchase_year_month'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m'))
    order_date_df['order_purchase_date'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m%d'))

    # Extracting attributes for purchase date - Day and Day of Week
    order_date_df['order_purchase_day'] = df['order_purchase_timestamp'].apply(lambda x: x.day)
    order_date_df['order_purchase_dayofweek'] = df['order_purchase_timestamp'].apply(lambda x: x.dayofweek)
    order_date_df['order_purchase_dayofweek_name'] = df['order_purchase_timestamp'].apply(lambda x: x.strftime('%a'))

    # Extracting attributes for purchase date - Hour and Time of the Day
    order_date_df['order_purchase_hour'] = df['order_purchase_timestamp'].apply(lambda x: x.hour)
    hours_bins = [-0.1, 6, 12, 18, 23]
    hours_labels = ['Dawn', 'Morning', 'Afternoon', 'Night']
    order_date_df['order_purchase_time_day'] = pd.cut(order_date_df['order_purchase_hour'], hours_bins, labels=hours_labels)

    order_date_df = order_date_df.reset_index()
    return order_date_df


def create_time_delivered_df(df):
    time_delivered_df = pd.DataFrame()
    # From datetime import datetime as dt
    time_delivered_df['diff_app_pur'] = (pd.to_datetime(df.order_approved_at) -
                                         pd.to_datetime(df.order_purchase_timestamp)).dt.seconds
    time_delivered_df['diff_car_app'] = (pd.to_datetime(df.order_delivered_carrier_date) -
                                         pd.to_datetime(df.order_approved_at)).dt.days
    time_delivered_df['diff_del_car'] = (pd.to_datetime(df.order_delivered_customer_date) -
                                         pd.to_datetime(df.order_delivered_carrier_date)).dt.days
    time_delivered_df['diff_est_act'] = (pd.to_datetime(df.order_estimated_delivery_date) -
                                         pd.to_datetime(df.order_delivered_customer_date)).dt.days

    time_delivered_df = time_delivered_df.reset_index()
    return time_delivered_df



def single_countplot(df, ax, x=None, y=None, top=None, order=True, hue=False, palette='plasma', width=0.75, sub_width=0.3, sub_size=12):
    # Checking for plotting by breaking down a categorical variable
    ncount = len(df)
    if x:
        col = x
    else:
        col = y

    # Checking for plotting the top categories
    if top is not None:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]

    # Validating other arguments and plotting the graph
    if hue != False:
        if order:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, order=df[col].value_counts().index, hue=hue)
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, hue=hue)
    else:
        if order:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, order=df[col].value_counts().index)
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax)

    # Formatting axes
    format_spines(ax, right_border=False)

    # Adding percentage labels
    if x:
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{}\n{:.1f}%'.format(int(y), 100. * y / ncount), (x.mean(), y), ha='center', va='bottom')
    else:
        for p in ax.patches:
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            ax.annotate('{} ({:.1f}%)'.format(int(x), 100. * x / ncount), (x, y.mean()), va='center')

def format_spines(ax, right_border=True):
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')

# make filter components
min_date = df["order_purchase_timestamp"].min()
max_date = df["order_purchase_timestamp"].max()

# ----- SIDEBAR -----
with st.sidebar:
    st.sidebar.header("Choose the Filter:")

    start_date, end_date = st.date_input(
        label="Date Filter", min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    review_score = st.sidebar.multiselect(
        "Select the Review Score:",
        options=df['review_score'].unique(),
        default=df['review_score'].unique())

    payment_type = st.sidebar.multiselect(
        "Select the Payment Type:",
        options=df['payment_type'].unique(),
        default=df['payment_type'].unique())

# Connect filter with main_df
df_selection = df[
    (df["order_purchase_timestamp"] >= str(start_date)) &
    (df["order_purchase_timestamp"] <= str(end_date))
    ]

df_selection = df_selection.query("review_score == @review_score & payment_type ==@payment_type")

if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

# assign main_df to helper functions
order_date_df = create_order_date_df(df_selection)
time_delivered_df = create_time_delivered_df(df_selection)



# ----- MAINPAGE -----
st.title(":bar_chart: Brazilian E-Commerce Data Analysis Dashboard")
st.markdown("###")

col1, col2, col3 = st.columns(3)

with col1:
    total_all_rides = time_delivered_df.diff_app_pur.mean()
    st.metric("Approval Time Average", value="{:.1f} sec".format(total_all_rides))
with col2:
    total_casual_rides = time_delivered_df.diff_car_app.mean()
    st.metric("Logistics Handover Duration", value="{:.1f} days".format(total_casual_rides))
with col3:
    total_registered_rides = time_delivered_df.diff_est_act.mean()
    st.metric("Delivery Delay Duration", value="{:.1f} days".format(total_registered_rides))

st.markdown("---")

# ----- CHART -----

# Create a line chart
# Create charts in new column
c1, c2 = st.columns(2)
with c1:
    df_fig_product_sales = df_selection.groupby(by=["product_id"])[["customer_id"]].count().sort_values(by="customer_id").tail(5)
    fig_product_sales = px.bar(
        df_fig_product_sales,
        x="customer_id",
        y=df_fig_product_sales.index,
        orientation="h",
        title="<b>Top 5 Products</b>",
        color_discrete_sequence=["#0083B8"] * len(df_fig_product_sales),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    c1.plotly_chart(fig_product_sales, use_container_width=True)
with c2:
    df_fig_category = df_selection.groupby(by=["product_category_name_x"])[["customer_id"]].count().sort_values(by="customer_id")
    fig_category = px.bar(
        df_fig_category,
        x=df_fig_category.index,
        y="customer_id",
        title="<b>Top Category Products</b>",
        color_discrete_sequence=["#0083B8"] * len(df_fig_category),
        template="plotly_white",
    )
    fig_category.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    c2.plotly_chart(fig_category, use_container_width=True)


st.markdown('**Evolution of Total Orders in Brazilian E-Commerce**')

# Lineplot - Evolution of e-commerce orders along time
fig, ax = plt.subplots(figsize=(20, 6))
sns.lineplot(data=order_date_df['order_purchase_year_month'].value_counts().sort_index(), ax=ax,
             color='darkslateblue', linewidth=2)
format_spines(ax, right_border=False)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
st.pyplot(fig)


# Create another charts in new column
c1, c2 = st.columns(2)
with c1:
    st.markdown('**Geolocation Orders**')
    lat = df_selection['geolocation_lat']
    lon = df_selection['geolocation_lng']

    fig1, ax1 = plt.subplots(figsize=(5, 15))

    m = Basemap(ax=ax1, llcrnrlat=-55.401805, llcrnrlon=-92.269176, urcrnrlat=13.884615, urcrnrlon=-27.581676)
    m.bluemarble()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
    m.drawcountries()
    m.scatter(lon, lat, zorder=10, alpha=0.5, color='tomato')
    st.pyplot(fig1)

# Column 2
with c2:
    st.markdown('**Total Orders by Time of the Day**')
    # Barchart - Total of orders by time of the day
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    day_color_list = ['darkslateblue', 'deepskyblue', 'darkorange', 'purple']
    single_countplot(order_date_df, x='order_purchase_time_day', ax=ax2, order=False, palette='YlGnBu')
    st.pyplot(fig2)
    ###
    st.markdown('**Total Orders by Day of Weekend**')
    # Barchart - Total of orders by day of week
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    single_countplot(order_date_df, x='order_purchase_dayofweek', ax=ax3, order=False, palette='YlGnBu')
    weekday_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax3.set_xticklabels(weekday_label)
    st.pyplot(fig3)


# ----- HIDE STREAMLIT STYLE -----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)