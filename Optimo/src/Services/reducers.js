const initialState = {
    portfolio: {}
  }
  
  export default (state = initialState, action) => {
    switch (action.type) {
      case 'UPDATE_PORTFOLIO':
        return {
          ...state,
          portfolio: action.payload
        }
      default:
        return state;
    }
  }
  