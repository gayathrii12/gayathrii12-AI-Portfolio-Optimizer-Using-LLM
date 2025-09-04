import React from 'react';
import styled from 'styled-components';
import { StatusBadgeProps } from '../../types';

const Badge = styled.span<{ status: string; size: string }>`
  display: inline-flex;
  align-items: center;
  padding: ${props => {
    switch (props.size) {
      case 'small': return '2px 8px';
      case 'large': return '8px 16px';
      default: return '4px 12px';
    }
  }};
  border-radius: 20px;
  font-size: ${props => {
    switch (props.size) {
      case 'small': return '0.75rem';
      case 'large': return '0.875rem';
      default: return '0.75rem';
    }
  }};
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  
  ${props => {
    switch (props.status) {
      case 'HEALTHY':
        return `
          color: #38a169;
          background-color: #f0fff4;
          border: 1px solid #9ae6b4;
        `;
      case 'WARNING':
        return `
          color: #d69e2e;
          background-color: #fffbeb;
          border: 1px solid #fbd38d;
        `;
      case 'CRITICAL':
        return `
          color: #e53e3e;
          background-color: #fed7d7;
          border: 1px solid #feb2b2;
        `;
      default:
        return `
          color: #718096;
          background-color: #f7fafc;
          border: 1px solid #e2e8f0;
        `;
    }
  }}
`;

const StatusIcon = styled.span`
  margin-right: 4px;
  font-size: 0.875em;
`;

const StatusBadge: React.FC<StatusBadgeProps> = ({ 
  status, 
  size = 'medium' 
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'HEALTHY':
        return '✅';
      case 'WARNING':
        return '⚠️';
      case 'CRITICAL':
        return '🚨';
      default:
        return '❓';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'HEALTHY':
        return 'Healthy';
      case 'WARNING':
        return 'Warning';
      case 'CRITICAL':
        return 'Critical';
      default:
        return 'Unknown';
    }
  };

  return (
    <Badge status={status} size={size}>
      <StatusIcon>{getStatusIcon(status)}</StatusIcon>
      {getStatusText(status)}
    </Badge>
  );
};

export default StatusBadge;