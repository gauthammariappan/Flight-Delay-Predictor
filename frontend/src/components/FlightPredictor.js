import React, { useState, useEffect } from 'react';
import { predictFlightDelay, checkHealth } from '../services/api';
import './FlightPredictor.css';

const FlightPredictor = () => {
  const today = new Date().toISOString().split('T')[0];
  
  const [formData, setFormData] = useState({
    origin: 'JFK',
    dest: 'LAX',
    date: today,
    hour: 8,
    minute: 0,
    period: 'AM'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('disconnected'));
  }, []);

  const airports = [
    'JFK', 'LAX', 'ORD', 'DFW', 'DEN', 'ATL', 'SFO', 'SEA', 'LAS', 'MCO', 
    'EWR', 'BOS', 'CLT', 'MIA', 'PHX', 'IAH', 'MSP', 'DTW', 'PHL', 'LGA',
    'BWI', 'SLC', 'DCA', 'SAN', 'TPA', 'PDX', 'HNL', 'BNA', 'AUS', 'MDW',
    'DAL', 'OAK', 'SJC', 'RDU', 'MSY', 'SAT', 'RSW', 'PIT', 'CVG', 'CMH',
    'IND', 'MCI', 'BUF', 'JAX', 'BDL', 'ONT', 'BUR', 'SMF', 'SNA', 'ALB', 'ABQ'
  ];

  const hours = Array.from({ length: 12 }, (_, i) => i + 1);
  const minutes = [0, 15, 30, 45];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'hour' || name === 'minute' ? parseInt(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictFlightDelay(formData);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to get prediction. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch(level) {
      case 'Low': return '#10b981';
      case 'Medium': return '#f59e0b';
      case 'High': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="predictor-container">
      <div className="predictor-card">
        <div className="header">
          <h1>✈️ Flight Delay Predictor</h1>
          <p className="subtitle">ML-powered risk assessment using XGBoost</p>
          <div className={`api-status ${apiStatus}`}>
            <span className="status-dot"></span>
            {apiStatus === 'connected' ? 'API Connected' : apiStatus === 'checking' ? 'Checking...' : 'API Disconnected'}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-grid">
            <div className="form-group">
              <label>Origin Airport</label>
              <select name="origin" value={formData.origin} onChange={handleInputChange}>
                {airports.map(airport => (
                  <option key={airport} value={airport}>{airport}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Destination Airport</label>
              <select name="dest" value={formData.dest} onChange={handleInputChange}>
                {airports.map(airport => (
                  <option key={airport} value={airport}>{airport}</option>
                ))}
              </select>
            </div>

            <div className="form-group full-width">
              <label>Flight Date</label>
              <input
                type="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                min={today}
              />
            </div>

            <div className="form-group">
              <label>Hour</label>
              <select name="hour" value={formData.hour} onChange={handleInputChange}>
                {hours.map(hour => (
                  <option key={hour} value={hour}>{hour}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Minute</label>
              <select name="minute" value={formData.minute} onChange={handleInputChange}>
                {minutes.map(min => (
                  <option key={min} value={min}>{min.toString().padStart(2, '0')}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>AM/PM</label>
              <select name="period" value={formData.period} onChange={handleInputChange}>
                <option value="AM">AM</option>
                <option value="PM">PM</option>
              </select>
            </div>
          </div>

          <button type="submit" className="predict-button" disabled={loading || apiStatus !== 'connected'}>
            {loading ? 'Analyzing...' : 'Predict Risk'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <div className="result-header">
              <h2>Prediction Results</h2>
            </div>
            
            <div className="risk-score-container">
              <div className="risk-score" style={{ borderColor: getRiskColor(prediction.riskLevel) }}>
                <div className="score-value" style={{ color: getRiskColor(prediction.riskLevel) }}>
                  {prediction.riskScore}%
                </div>
                <div className="risk-label" style={{ color: getRiskColor(prediction.riskLevel) }}>
                  {prediction.riskLevel} Risk
                </div>
              </div>
            </div>

            <div className="result-details">
              <div className="detail-item">
                <span className="detail-label">Status:</span>
                <span className={`detail-value ${prediction.willBeDelayed ? 'delayed' : 'on-time'}`}>
                  {prediction.willBeDelayed ? 'Likely Delayed/Cancelled' : 'Likely On Time'}
                </span>
              </div>
              
              {prediction.calculatedDistance && (
                <div className="detail-item">
                  <span className="detail-label">Flight Distance:</span>
                  <span className="detail-value">{prediction.calculatedDistance} miles</span>
                </div>
              )}
              
              <div className="recommendation">
                <strong>Recommendation:</strong>
                <p>{prediction.recommendation}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>Built with React + XGBoost | Trained on 1M+ flight records</p>
      </footer>
    </div>
  );
};

export default FlightPredictor;