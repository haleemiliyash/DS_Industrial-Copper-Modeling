import numpy as np
import pickle
import json
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Copper Price model', layout="wide")
st.title('DS_Industrial Copper Model By Abdul Haleem')


with open("D:/project/.venv/Industrial_copper_modeling/price_model_pkl",'rb') as file:
    price_model = pickle.load(file)
with open("D:/project/.venv/Industrial_copper_modeling/status_model_pkl",'rb') as file:
    status_model = pickle.load(file)
     
def home():
    st.markdown(
          "<h1 style='text-align: center; color: pink;'>Copper Modeling capstone</h1>",
          unsafe_allow_html=True)

    st.header(' :green[Importance of Machine Learning in Copper Pricing Prediction]')
    st.write('### :blue[Introduction:]')
    st.write('Welcome to the Copper Modeling Web Page! This platform utilizes machine learning algorithms to provide valuable insights into copper pricing and transaction status.') 
    
    
    st.write('### :blue[Importance of Machine Learning:]')
    st.write('''In the manufacturing domain, accurately predicting prices and transaction outcomes is paramount.Traditional methods often struggle to account for the myriad of factors influencing copper prices and transaction statuses. This is where machine learning shines. By leveraging vast amounts of historical data and transaction statuses. This is where machine learning shines. By leveraging vast amounts of historical data and sophisticated algorithms, machine learning can uncover hidden patterns and make highly accurate predictions.''') 
    
    
      
          
    st.header(' :green[Use Case of This Project:]')
    st.write('Our project focuses on two key aspects:')

    st.write('### :violet[Selling Price Prediction:]')
    st.write('''Users can input various parameters related to copper transactions, and our machine learning model predicts the selling price. This empowers stakeholders to make informed.decisions about pricing strategies and negotiations''')
    
    st.write('### :violet[Status Prediction:]') 
    st.write('Users provide transaction details, and our model predicts whether the transaction is likely to be successful ("Won") or unsuccessful ("Lost").') 
    

#selling price prediction  model      
def sell_price():
      st.write(' :white[Fill the below details to find Predicted Selling Price]')
  #define dict for mapping
      status_dict = {'Won':1,'Draft':0,'To be approved':6,'Lost': 1,'Not lost for AM':2,'Wonderful':8,'Revised':5,'Offered':4,'Offerable':3}
      item_type_dict={'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
      

      country_val=[25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
      status_val=['Won','Draft','To be approved','Lost','Not lost for AM''Wonderful','Revised','Offered','Offerable']
      item_type_val = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
      application_val = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 
                            40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
      
      product_ref_val = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                            164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 
                          1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 
                          1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

      user_data = pd.DataFrame(columns=[['quantity tons','status','item_type','application','thickness','width','country', 'customer','product_ref']])

      #user input
      with st.form("Regression"):
          col1,col2,col3 = st.columns([0.5,0.2,0.5])

          with col1:
              quantity=st.number_input(label='Quantity tons',min_value=0.1,max_value=20.73,key=1)
              country=st.selectbox(label='Country', options=country_val)
              item_type=st.selectbox(label='Item type',options=item_type_dict)
              thickness=st.number_input(label='Thickness',min_value=0.1,max_value=7.82)
              product_ref = st.selectbox(label='Product Reference', options=product_ref_val)
          
          with col3:
              customer=st.number_input(label='Customer ID',min_value=30147616,max_value=30408185)
              status=st.selectbox(label='Status',options=status_val)
              application = st.selectbox(label='Application', options=application_val)
              width = st.number_input(label='Width', min_value=1, max_value=2990)
              st.write('')
              st.write('')
          with col2:
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.markdown('Click below button to predict')
              button=st.form_submit_button(label='predict price')

          col1,col2,col3 = st.columns([0.35,0.35,0.30])
          with col1:
            st.caption(body='*Min and Max values are reference only')

      if button:
          # covert status and item_type into encode values
        status_int = status_dict.get(status)
        item_type_int = item_type_dict.get(item_type)
         
        input_ar=np.array([[quantity,status_int,item_type_int,application,thickness,width,country,customer,product_ref]])

        Y_pred=price_model.predict(input_ar)
        sell_price = np.exp(Y_pred[0])
        sell_price=round(sell_price,2)
        st.header(f'Predicted Selling Price is: {sell_price}')

#status prediction model
def status_predict():
    st.write(' :white[Fill the below details to find Predict Status]')
  #define dict for mapping
    status_dict = {'Won':1,'Draft':0,'To be approved':6,'Lost': 1,'Not lost for AM':2,'Wonderful':8,'Revised':5,'Offered':4,'Offerable':3}
    item_type_dict={'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}
      

    country_val=[25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    status_val=['Won','Draft','To be approved','Lost','Not lost for AM''Wonderful','Revised','Offered','Offerable']
    item_type_val = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
    application_val = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 
                            40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
      
    product_ref_val = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                            164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 
                          1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 
                          1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

    user_data = pd.DataFrame(columns=[['quantity tons','status','item_type','application','thickness','width','country', 'customer','product_ref']])
    
    with st.form("Regression"):
        col1,col2,col3 = st.columns([0.5,0.2,0.5])
        with col1:
              quantity=st.number_input(label='Quantity tons',min_value=0.1,max_value=20.73,key=1)
              country=st.selectbox(label='Country', options=country_val)
              item_type=st.selectbox(label='Item type',options=item_type_dict)
              thickness=st.number_input(label='Thickness',min_value=0.1,max_value=7.82)
              product_ref = st.selectbox(label='Product Reference', options=product_ref_val)
          
        with col3:
              customer=st.number_input(label='Customer ID',min_value=30147616,max_value=30408185)
              selling_price = st.number_input(label='Selling_price', min_value=0 , max_value=2500)
              application = st.selectbox(label='Application', options=application_val)
              width = st.number_input(label='Width', min_value=1, max_value=2990)
              st.write('')
              st.write('')
        with col2:
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.write('')
              st.markdown('Click below button to predict')
              button=st.form_submit_button(label='predict status')

        col1,col2,col3 = st.columns([0.35,0.35,0.30])
        with col1:
            st.caption(body='*Min and Max values are reference only')
    if button:
        item_type_ip = item_type_dict.get(item_type)
        input_ar=np.array([[quantity,selling_price,item_type_ip ,application,thickness,width,country,customer,product_ref]])
        Y_pred=price_model.predict(input_ar)

        if Y_pred[0]==1:
            st.header('Predicted status is "Won"')
        else:
            st.header('Predicted status is "Loss"')

            

with st.sidebar:
  option = option_menu("Main menu",['Home','Selling Price Prediction','Status Prediction'],
                       icons=["house","cloud-upload","list-task","pencil-square"],
                       menu_icon="cast",
                       styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "green"},
                                   "nav-link-selected": {"background-color": "green"}},
                       default_index=0)
if option=='Home':
    home()
elif option=='Selling Price Prediction':
    sell_price()
elif option=='Status Prediction':
    status_predict()
    