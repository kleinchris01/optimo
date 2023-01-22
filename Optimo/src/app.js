import React from 'react';
import { Provider } from 'react-redux';
import { createStore } from 'redux';
import rootReducer from './store/reducers';
import Chart from './components/Chart';
import Portfolio from './components/Portfolio';

const store = createStore(rootReducer);

const App = () => (
  <Provider store={store}>
    <Chart />
    <Portfolio />
  </Provider>
);

export default App;

