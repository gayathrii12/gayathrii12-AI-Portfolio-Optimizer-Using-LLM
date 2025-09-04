import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { SystemStatus } from '../../types';
import StatusBadge from '../Common/StatusBadge';
import { apiService } from '../../services/api';

const HeaderContainer = styled.header`
  background: white;
  border-bottom: 1px solid #e2e8f0;
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
`;

const StatusSection = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
`;

const LastUpdated = styled.span`
  font-size: 0.875rem;
  color: #718096;
`;

const RefreshButton = styled.button`
  background: #3182ce;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background: #2c5aa0;
  }

  &:disabled {
    background: #a0aec0;
    cursor: not-allowed;
  }
`;

const Header: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>('HEALTHY');
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchSystemStatus = async () => {
    try {
      setIsRefreshing(true);
      const healthData = await apiService.getSystemHealth();
      setSystemStatus(healthData.system_status);
      
      // Handle timestamp properly to avoid Invalid Date
      if (healthData.timestamp) {
        const timestamp = new Date(healthData.timestamp);
        if (!isNaN(timestamp.getTime())) {
          setLastUpdated(timestamp.toLocaleTimeString());
        } else {
          setLastUpdated(new Date().toLocaleTimeString());
        }
      } else {
        setLastUpdated(new Date().toLocaleTimeString());
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      setSystemStatus('CRITICAL');
      setLastUpdated(''); // Clear invalid date on error
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchSystemStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchSystemStatus();
  };

  return (
    <HeaderContainer>
      <Title>S&P 500 Historical Analysis • Real Excel Data</Title>
      <StatusSection>
        <StatusBadge status={systemStatus} />
        {lastUpdated && lastUpdated !== 'Invalid Date' && (
          <LastUpdated>
            Last updated: {lastUpdated}
          </LastUpdated>
        )}
        <RefreshButton 
          onClick={handleRefresh} 
          disabled={isRefreshing}
        >
          {isRefreshing ? 'Refreshing...' : 'Refresh'}
        </RefreshButton>
      </StatusSection>
    </HeaderContainer>
  );
};

export default Header;