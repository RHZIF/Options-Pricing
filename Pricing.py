import streamlit as st
import pandas as pd 
import requests
import investpy as py
#from bs4 import BeautifulSoup
#import Casabourselib as cbl 
from enum import Enum
from datetime import datetime, timedelta
#from MonteCarloSimulation import MonteCarloPricing
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import math
from numpy import *
# Local package imports



class MonteCarloPricing():
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    We simulate underlying asset price on expiry date using random stochastic process - Brownian motion.
    For the simulation generated prices at maturity, we calculate and sum up their payoffs, average them and discount the final value.
    That value represents option price
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_simulations: number of potential random underlying price movements 
        """
        # Parameters for Brownian process
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma 

        # Parameters for simulation
        self.N = number_of_simulations
        self.num_of_steps = days_to_maturity
        self.dt = self.T / self.num_of_steps

    def simulate_prices(self):
        """
        Simulating price movement of underlying prices using Brownian random process.
        Saving random results.
        """
        np.random.seed(20)
        self.simulation_results = None

        # Initializing price movements for simulation: rows as time index and columns as different random price movements.
        S = np.zeros((self.num_of_steps, self.N))        
        # Starting value for all price movements is the current spot price
        S[0] = self.S_0

        for t in range(1, self.num_of_steps):
            # Random values to simulate Brownian motion (Gaussian distibution)
            Z = np.random.standard_normal(self.N)
            # Updating prices for next point in time 
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))

        self.simulation_results_S = S

    def _calculate_call_option_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    

    def _calculate_put_option_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))
       

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results_S[:,0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.num_of_steps, label='Strike Price')
        plt.xlim([0, self.num_of_steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        return plt 

#_______________________________
# Main title
#st.title('Pricing des Options')


#apptitle = 'Projet Pricing des options européennes'


# Title the app
st.title('Projet Pricing des options ')

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data

# @st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data
    
st.sidebar.markdown("## Selectioner la methode de calcule ")


st.markdown('### Travail realisé par: ')
st.markdown('- Ahmed Ouaboune            (Ahmed.Ouaboune@edu.uiz.ac.ma)')
st.markdown('- Yassine Rhzif             (Yassine.Rhzif@edu.uiz.ac.ma)')
st.markdown('#### Filière: finance et ingenierie decisionnelle')
st.markdown('<center><img src="https://raw.githubusercontent.com/RHZIF/streamlit_test/main/ensa.png" width="300"  height="100" alt="Ensa logo"></center>', unsafe_allow_html=True)
st.markdown("### Sous l'encadrement de Pr. Brahim El Asri")
st.markdown('##')
st.markdown('__________________________________________________________')


#dropdown = st.sidebar.selectbox("Choisir une action", py.get_stocks(country='morocco').name)
monte = st.sidebar.selectbox("methode de calcule ", ['classique','reduction de la variance'])

#-----------------------

if monte == "classique":
  # Parameters for Monte Carlo simulation
    #ticker = st.text_input('Ticker symbol', 'AAPL')
    ticker = st.selectbox("Ticker symbol", py.get_stocks_list(country='United States'))
    strike_price = st.number_input('Strike price', 100)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)
    data = py.get_stock_historical_data(stock=ticker, country='United states', from_date='01/01/2021', to_date='31/12/2021')
    data['Log returns'] = np.log(data['Close']/data['Close'].shift())
    volatility = data['Log returns'].std()*252**.5
    st.write('Sigma = ', round(volatility, 2)*100, '%')


    if st.button(f'Calculate option price for {ticker}'):
        # Getting data for selected ticker
        data = py.get_stock_historical_data(stock=ticker, country='United States', from_date="01/01/1900", to_date= datetime.today().strftime('%d/%m/%Y'))
        st.write(data.tail())
        st.line_chart(data.Close)

        # Formating simulation parameters
        spot_price = data.Close[-1] 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        # ESimulating stock movements
        MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
        MC.simulate_prices()

        # Visualizing Monte Carlo Simulation
        plt = MC.plot_simulation_results(num_of_movements)
        st.pyplot(plt)

        # Calculating call/put option price
        call_option_price = MC._calculate_call_option_price()
        put_option_price = MC._calculate_put_option_price()

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')

else:

    # Parameters for Monte Carlo simulation
    #ticker = st.text_input('Ticker symbol', 'AAPL')
    ticker = st.selectbox("Ticker symbol", py.get_stocks_list(country='United States'))
    strike_price = st.number_input('Strike price', 100)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)
    if st.button(f'Calculate option price for {ticker}'):
        # Getting data for selected ticker
        data = py.get_stock_historical_data(stock=ticker, country='United States', from_date="01/01/1900", to_date= datetime.today().strftime('%d/%m/%Y'))
        st.write(data.tail())
        st.line_chart(data.Close)

        # Formating simulation parameters
        spot_price = data.Close[-1] 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        # ESimulating stock movements
        MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
        MC.simulate_prices()

        # Visualizing Monte Carlo Simulation
        plt = MC.plot_simulation_results(num_of_movements)
        st.pyplot(plt)
        #___________________________
        S0 = spot_price ; K = strike_price ; T= days_to_maturity; r = risk_free_rate; sigma = sigma
        M = 365; dt = T / M; I = number_of_simulations
        S1 = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt + sigma *math.sqrt(dt)* random.standard_normal((M + 1,I)), axis=0))
        S2 = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt + sigma *(-math.sqrt(dt)* random.standard_normal((M + 1,I))), axis=0))
        S1[0] = S0
        S2[0] = S0
        # Estimation de monte carlo 
        #call
        C0_call1 = math.exp(-r * T) * sum(maximum(S1[-1] - K, 0)) / I 
        C0_call2 = math.exp(-r * T) * sum(maximum(S2[-1] - K, 0)) / I
        #C0_call = math.exp(-r * T) *((sum(maximum(S1[-1] - K, 0))+sum(maximum(S2[-1] - K, 0)))/2)/I
        C0_call = (C0_call1+C0_call2)/2
        #put
        C0_put1 = math.exp(-r * T) * sum(maximum(K- S1[-1] , 0)) / I 
        C0_put2 = math.exp(-r * T) * sum(maximum(K- S2[-1], 0)) / I
        #C0_put = math.exp(-r * T) *((sum(maximum(K - S1[-1] , 0))+sum(maximum(K - S2[-1], 0)))/2)/I
        C0_put = (C0_put2+C0_put1)/2
        #________________________________________
        # Calculating call/put option price
        call_option_price = C0_call
        put_option_price  = C0_put
        #call_option_price = std((math.exp(-r * T) * maximum(S2[-1] - strike_price, 0) +math.exp(-r * T) * maximum(S1[-1] - strike_price , 0) )/2)/math.sqrt(I)
        #put_option_price = std((math.exp(-r * T) * maximum(strike_price- S2[-1], 0) +math.exp(-r * T) * maximum(strike_price- S1[-1] , 0) )/2)/math.sqrt(I))

        # Displaying call/put option price
        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')
        st.subheader(f'Call option error: {std((math.exp(-r * T) * maximum(S2[-1] - K, 0) +math.exp(-r * T) * maximum(S1[-1] - K , 0) )/2)/math.sqrt(I)}')
        st.subheader(f'Put option error: {std((math.exp(-r * T) * maximum(K- S2[-1], 0) +math.exp(-r * T) * maximum(K- S1[-1] , 0) )/2)/math.sqrt(I)}')


#------------------





