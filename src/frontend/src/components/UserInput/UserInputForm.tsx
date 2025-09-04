import React, { useState } from 'react';
import styled from 'styled-components';

// Types for user input
export interface UserInputData {
  investment_amount: number;
  investment_type: 'lumpsum' | 'sip';
  tenure_years: number;
  risk_profile: 'Low' | 'Moderate' | 'High';
  return_expectation: number;
  monthly_amount?: number; // For SIP
}

interface UserInputFormProps {
  onSubmit: (data: UserInputData) => void;
  loading?: boolean;
}

const FormContainer = styled.div`
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h2`
  color: #1a202c;
  margin-bottom: 1.5rem;
  text-align: center;
  font-size: 1.5rem;
  font-weight: 600;
`;

const FormGroup = styled.div`
  margin-bottom: 1.5rem;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  color: #2d3748;
  font-weight: 500;
  font-size: 0.9rem;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;

  &:focus {
    outline: none;
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
  }

  &:invalid {
    border-color: #e53e3e;
  }
`;

const Select = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  background: white;
  transition: border-color 0.2s;

  &:focus {
    outline: none;
    border-color: #3182ce;
    box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1);
  }
`;

const RadioGroup = styled.div`
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
`;

const RadioOption = styled.label`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  padding: 0.5rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  transition: all 0.2s;

  &:hover {
    border-color: #cbd5e0;
  }

  input:checked + & {
    border-color: #3182ce;
    background-color: #ebf8ff;
  }
`;

const RadioInput = styled.input`
  margin: 0;
`;

const SubmitButton = styled.button`
  width: 100%;
  padding: 1rem;
  background: #3182ce;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover:not(:disabled) {
    background: #2c5282;
  }

  &:disabled {
    background: #a0aec0;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  font-size: 0.875rem;
  margin-top: 0.25rem;
`;

const HelpText = styled.div`
  color: #718096;
  font-size: 0.875rem;
  margin-top: 0.25rem;
`;

const UserInputForm: React.FC<UserInputFormProps> = ({ onSubmit, loading = false }) => {
  const [formData, setFormData] = useState<UserInputData>({
    investment_amount: 100000,
    investment_type: 'lumpsum',
    tenure_years: 10,
    risk_profile: 'Moderate',
    return_expectation: 12,
  });

  const [errors, setErrors] = useState<Partial<Record<keyof UserInputData, string>>>({});

  const validateForm = (): boolean => {
    const newErrors: Partial<Record<keyof UserInputData, string>> = {};

    // Investment amount validation
    if (!formData.investment_amount || formData.investment_amount <= 0) {
      newErrors.investment_amount = 'Investment amount must be greater than 0';
    } else if (formData.investment_amount > 1000000000) {
      newErrors.investment_amount = 'Investment amount exceeds maximum limit';
    }

    // Tenure validation
    if (!formData.tenure_years || formData.tenure_years < 1 || formData.tenure_years > 50) {
      newErrors.tenure_years = 'Tenure must be between 1 and 50 years';
    }

    // Return expectation validation
    if (!formData.return_expectation || formData.return_expectation < 0 || formData.return_expectation > 50) {
      newErrors.return_expectation = 'Return expectation must be between 0% and 50%';
    }

    // SIP monthly amount validation
    if (formData.investment_type === 'sip') {
      if (!formData.monthly_amount || formData.monthly_amount <= 0) {
        newErrors.monthly_amount = 'Monthly SIP amount must be greater than 0';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const handleInputChange = (field: keyof UserInputData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: undefined
      }));
    }
  };

  return (
    <FormContainer>
      <Title>Investment Planning Calculator</Title>
      
      <form onSubmit={handleSubmit}>
        <FormGroup>
          <Label>Investment Type</Label>
          <RadioGroup>
            <RadioOption>
              <RadioInput
                type="radio"
                name="investment_type"
                value="lumpsum"
                checked={formData.investment_type === 'lumpsum'}
                onChange={(e) => handleInputChange('investment_type', e.target.value)}
              />
              Lump Sum Investment
            </RadioOption>
            <RadioOption>
              <RadioInput
                type="radio"
                name="investment_type"
                value="sip"
                checked={formData.investment_type === 'sip'}
                onChange={(e) => handleInputChange('investment_type', e.target.value)}
              />
              Systematic Investment Plan (SIP)
            </RadioOption>
          </RadioGroup>
        </FormGroup>

        <FormGroup>
          <Label>
            {formData.investment_type === 'lumpsum' ? 'Investment Amount ($)' : 'Initial Investment ($)'}
          </Label>
          <Input
            type="number"
            min="1"
            max="1000000000"
            step="1000"
            value={formData.investment_amount}
            onChange={(e) => handleInputChange('investment_amount', parseFloat(e.target.value) || 0)}
            placeholder="Enter investment amount"
          />
          {errors.investment_amount && <ErrorMessage>{errors.investment_amount}</ErrorMessage>}
          <HelpText>
            {formData.investment_type === 'lumpsum' 
              ? 'Total amount to invest at once'
              : 'Initial lump sum amount (optional for SIP)'
            }
          </HelpText>
        </FormGroup>

        {formData.investment_type === 'sip' && (
          <FormGroup>
            <Label>Monthly SIP Amount ($)</Label>
            <Input
              type="number"
              min="1"
              step="100"
              value={formData.monthly_amount || ''}
              onChange={(e) => handleInputChange('monthly_amount', parseFloat(e.target.value) || 0)}
              placeholder="Enter monthly SIP amount"
            />
            {errors.monthly_amount && <ErrorMessage>{errors.monthly_amount}</ErrorMessage>}
            <HelpText>Amount to invest every month</HelpText>
          </FormGroup>
        )}

        <FormGroup>
          <Label>Investment Tenure (Years)</Label>
          <Input
            type="number"
            min="1"
            max="50"
            value={formData.tenure_years}
            onChange={(e) => handleInputChange('tenure_years', parseInt(e.target.value) || 0)}
            placeholder="Enter investment tenure"
          />
          {errors.tenure_years && <ErrorMessage>{errors.tenure_years}</ErrorMessage>}
          <HelpText>How long do you plan to stay invested?</HelpText>
        </FormGroup>

        <FormGroup>
          <Label>Risk Profile</Label>
          <Select
            value={formData.risk_profile}
            onChange={(e) => handleInputChange('risk_profile', e.target.value)}
          >
            <option value="Low">Low Risk (Conservative)</option>
            <option value="Moderate">Moderate Risk (Balanced)</option>
            <option value="High">High Risk (Aggressive)</option>
          </Select>
          <HelpText>
            {formData.risk_profile === 'Low' && 'Conservative approach with focus on capital preservation'}
            {formData.risk_profile === 'Moderate' && 'Balanced approach with moderate growth potential'}
            {formData.risk_profile === 'High' && 'Aggressive approach with higher growth potential'}
          </HelpText>
        </FormGroup>

        <FormGroup>
          <Label>Expected Annual Return (%)</Label>
          <Input
            type="number"
            min="0"
            max="50"
            step="0.5"
            value={formData.return_expectation}
            onChange={(e) => handleInputChange('return_expectation', parseFloat(e.target.value) || 0)}
            placeholder="Enter expected return"
          />
          {errors.return_expectation && <ErrorMessage>{errors.return_expectation}</ErrorMessage>}
          <HelpText>
            Realistic expectations: Conservative (6-8%), Moderate (8-12%), Aggressive (12-15%)
          </HelpText>
        </FormGroup>

        <SubmitButton type="submit" disabled={loading}>
          {loading ? 'Generating Portfolio...' : 'Generate Portfolio Recommendation'}
        </SubmitButton>
      </form>
    </FormContainer>
  );
};

export default UserInputForm;