import React, { useEffect, useState } from 'react';
import { useDispatch } from 'react-redux';
import { updatePortfolio } from '../store/actions';
import api from '../services/api';

const Portfolio = () => {
  const [riskTolerance, setRiskTolerance] = useState(0.5);
  const dispatch = useDispatch();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = await api.post('/data', { riskTolerance });
    dispatch(updatePortfolio(data));
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          value={riskTolerance}
          onChange={e => setRiskTolerance(e.target.value)}
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}

export default Portfolio;
