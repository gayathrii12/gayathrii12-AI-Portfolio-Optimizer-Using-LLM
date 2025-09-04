import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import UserInputForm, { UserInputData } from '../UserInputForm';

describe('UserInputForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  const renderForm = (props = {}) => {
    return render(
      <UserInputForm
        onSubmit={mockOnSubmit}
        {...props}
      />
    );
  };

  test('renders form with all required fields', () => {
    renderForm();

    expect(screen.getByText('Investment Planning Calculator')).toBeInTheDocument();
    expect(screen.getByLabelText(/investment type/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/investment amount/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/investment tenure/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/risk profile/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/expected annual return/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate portfolio recommendation/i })).toBeInTheDocument();
  });

  test('has correct default values', () => {
    renderForm();

    expect(screen.getByDisplayValue('100000')).toBeInTheDocument(); // investment amount
    expect(screen.getByDisplayValue('10')).toBeInTheDocument(); // tenure
    expect(screen.getByDisplayValue('12')).toBeInTheDocument(); // return expectation
    expect(screen.getByDisplayValue('Moderate')).toBeInTheDocument(); // risk profile
    expect(screen.getByRole('radio', { name: /lump sum investment/i })).toBeChecked();
  });

  test('shows monthly SIP field when SIP is selected', async () => {
    renderForm();

    // Initially, monthly SIP field should not be visible
    expect(screen.queryByLabelText(/monthly sip amount/i)).not.toBeInTheDocument();

    // Select SIP option
    fireEvent.click(screen.getByRole('radio', { name: /systematic investment plan/i }));

    // Monthly SIP field should now be visible
    await waitFor(() => {
      expect(screen.getByLabelText(/monthly sip amount/i)).toBeInTheDocument();
    });
  });

  test('validates required fields', async () => {
    renderForm();

    // Clear investment amount
    const amountInput = screen.getByLabelText(/investment amount/i);
    fireEvent.change(amountInput, { target: { value: '0' } });

    // Try to submit
    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/investment amount must be greater than 0/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('validates investment amount limits', async () => {
    renderForm();

    // Set amount above limit
    const amountInput = screen.getByLabelText(/investment amount/i);
    fireEvent.change(amountInput, { target: { value: '2000000000' } }); // 2 billion

    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/investment amount exceeds maximum limit/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('validates tenure range', async () => {
    renderForm();

    // Set tenure below minimum
    const tenureInput = screen.getByLabelText(/investment tenure/i);
    fireEvent.change(tenureInput, { target: { value: '0' } });

    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/tenure must be between 1 and 50 years/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('validates return expectation range', async () => {
    renderForm();

    // Set return expectation above limit
    const returnInput = screen.getByLabelText(/expected annual return/i);
    fireEvent.change(returnInput, { target: { value: '60' } });

    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/return expectation must be between 0% and 50%/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('validates monthly SIP amount when SIP is selected', async () => {
    renderForm();

    // Select SIP
    fireEvent.click(screen.getByRole('radio', { name: /systematic investment plan/i }));

    // Wait for SIP field to appear, then submit without filling it
    await waitFor(() => {
      expect(screen.getByLabelText(/monthly sip amount/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/monthly sip amount must be greater than 0/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('submits form with valid lump sum data', async () => {
    renderForm();

    // Fill form with valid data
    const amountInput = screen.getByLabelText(/investment amount/i);
    fireEvent.change(amountInput, { target: { value: '150000' } });

    const tenureInput = screen.getByLabelText(/investment tenure/i);
    fireEvent.change(tenureInput, { target: { value: '15' } });

    const returnInput = screen.getByLabelText(/expected annual return/i);
    fireEvent.change(returnInput, { target: { value: '10' } });

    // Select High risk
    const riskSelect = screen.getByLabelText(/risk profile/i);
    fireEvent.change(riskSelect, { target: { value: 'High' } });

    // Submit form
    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        investment_amount: 150000,
        investment_type: 'lumpsum',
        tenure_years: 15,
        risk_profile: 'High',
        return_expectation: 10
      });
    });
  });

  test('submits form with valid SIP data', async () => {
    renderForm();

    // Select SIP
    fireEvent.click(screen.getByRole('radio', { name: /systematic investment plan/i }));

    // Wait for SIP fields to appear
    await waitFor(() => {
      expect(screen.getByLabelText(/monthly sip amount/i)).toBeInTheDocument();
    });

    // Fill SIP-specific fields
    const monthlyAmountInput = screen.getByLabelText(/monthly sip amount/i);
    fireEvent.change(monthlyAmountInput, { target: { value: '5000' } });

    // Modify other fields
    const amountInput = screen.getByLabelText(/initial investment/i);
    fireEvent.change(amountInput, { target: { value: '50000' } });

    // Select Low risk
    const riskSelect = screen.getByLabelText(/risk profile/i);
    fireEvent.change(riskSelect, { target: { value: 'Low' } });

    // Submit form
    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        investment_amount: 50000,
        investment_type: 'sip',
        tenure_years: 10,
        risk_profile: 'Low',
        return_expectation: 12,
        monthly_amount: 5000
      });
    });
  });

  test('shows loading state when loading prop is true', () => {
    renderForm({ loading: true });

    const submitButton = screen.getByRole('button');
    expect(submitButton).toHaveTextContent('Generating Portfolio...');
    expect(submitButton).toBeDisabled();
  });

  test('clears field errors when user starts typing', async () => {
    renderForm();

    // Create an error by submitting invalid data
    const amountInput = screen.getByLabelText(/investment amount/i);
    fireEvent.change(amountInput, { target: { value: '0' } });

    fireEvent.click(screen.getByRole('button', { name: /generate portfolio recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/investment amount must be greater than 0/i)).toBeInTheDocument();
    });

    // Start typing in the field
    fireEvent.change(amountInput, { target: { value: '1' } });

    // Error should be cleared
    await waitFor(() => {
      expect(screen.queryByText(/investment amount must be greater than 0/i)).not.toBeInTheDocument();
    });
  });

  test('shows appropriate help text for different risk profiles', async () => {
    renderForm();

    // Test Moderate (default)
    expect(screen.getByText(/balanced approach with moderate growth potential/i)).toBeInTheDocument();

    // Test Low risk
    const riskSelect = screen.getByLabelText(/risk profile/i);
    fireEvent.change(riskSelect, { target: { value: 'Low' } });
    await waitFor(() => {
      expect(screen.getByText(/conservative approach with focus on capital preservation/i)).toBeInTheDocument();
    });

    // Test High risk
    fireEvent.change(riskSelect, { target: { value: 'High' } });
    await waitFor(() => {
      expect(screen.getByText(/aggressive approach with higher growth potential/i)).toBeInTheDocument();
    });
  });

  test('updates field labels when investment type changes', async () => {
    renderForm();

    // Initially should show "Investment Amount"
    expect(screen.getByLabelText(/investment amount \(\$\)/i)).toBeInTheDocument();

    // Switch to SIP
    fireEvent.click(screen.getByRole('radio', { name: /systematic investment plan/i }));

    // Should now show "Initial Investment"
    await waitFor(() => {
      expect(screen.getByLabelText(/initial investment \(\$\)/i)).toBeInTheDocument();
    });
  });
});