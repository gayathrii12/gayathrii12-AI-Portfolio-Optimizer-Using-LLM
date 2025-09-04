import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { DataQualitySummary } from '../types';
import { apiService } from '../services/api';
import MetricCard from '../components/Common/MetricCard';
import BarChart from '../components/Charts/BarChart';
import LoadingSpinner from '../components/Common/LoadingSpinner';

const PageContainer = styled.div`
  padding: 20px;
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 30px;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const Card = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 20px;
`;

const CardTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 15px;
`;

const DatasetCard = styled(Card)<{ qualityScore: number }>`
  border-left: 4px solid ${props => 
    props.qualityScore >= 90 ? '#38a169' :
    props.qualityScore >= 70 ? '#d69e2e' : '#e53e3e'
  };
`;

const QualityScore = styled.div<{ score: number }>`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => 
    props.score >= 90 ? '#38a169' :
    props.score >= 70 ? '#d69e2e' : '#e53e3e'
  };
  margin-bottom: 8px;
`;

const DatasetName = styled.h4`
  font-size: 1rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 10px;
`;

const IssuesList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 10px 0 0;
`;

const IssueItem = styled.li`
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 0.875rem;
  color: #4a5568;
`;

const IssueCount = styled.span<{ count: number }>`
  font-weight: 600;
  color: ${props => props.count > 0 ? '#e53e3e' : '#38a169'};
`;

const ErrorMessage = styled.div`
  color: #e53e3e;
  background-color: #fed7d7;
  border: 1px solid #feb2b2;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
`;

const DataQuality: React.FC = () => {
  const [qualityData, setQualityData] = useState<DataQualitySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchQualityData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getDataQualitySummary();
      setQualityData(data);
    } catch (err) {
      setError('Failed to load data quality information');
      console.error('Data quality fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQualityData();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchQualityData, 60000);
    
    return () => clearInterval(interval);
  }, []);

  if (loading && !qualityData) {
    return (
      <PageContainer>
        <PageTitle>Data Quality Monitoring</PageTitle>
        <LoadingSpinner size="large" />
      </PageContainer>
    );
  }

  if (error && !qualityData) {
    return (
      <PageContainer>
        <PageTitle>Data Quality Monitoring</PageTitle>
        <ErrorMessage>{error}</ErrorMessage>
      </PageContainer>
    );
  }

  if (!qualityData) {
    return (
      <PageContainer>
        <PageTitle>Data Quality Monitoring</PageTitle>
        <ErrorMessage>No data quality information available</ErrorMessage>
      </PageContainer>
    );
  }

  // Prepare chart data with null checks
  const datasets = qualityData.datasets ? Object.keys(qualityData.datasets) : [];
  const qualityScores = datasets.map(dataset => 
    qualityData.datasets && qualityData.datasets[dataset] ? qualityData.datasets[dataset].quality_score : 0
  );
  const completenessScores = datasets.map(dataset => 
    qualityData.datasets && qualityData.datasets[dataset] ? qualityData.datasets[dataset].completeness : 0
  );
  
  const qualityScoreData = {
    labels: datasets.map(name => name.replace(/_/g, ' ')),
    datasets: [
      {
        label: 'Quality Score (%)',
        data: qualityScores,
        backgroundColor: qualityScores.map(score => 
          score >= 90 ? '#38a169' :
          score >= 70 ? '#d69e2e' : '#e53e3e'
        ),
        borderWidth: 1
      }
    ]
  };

  const completenessData = {
    labels: datasets.map(name => name.replace(/_/g, ' ')),
    datasets: [
      {
        label: 'Data Completeness (%)',
        data: completenessScores,
        backgroundColor: '#3182ce',
        borderColor: '#2c5aa0',
        borderWidth: 1
      }
    ]
  };

  const issuesData = {
    labels: datasets.map(name => name.replace(/_/g, ' ')),
    datasets: [
      {
        label: 'Missing Values',
        data: datasets.map(dataset => qualityData.datasets[dataset].issues.missing_values),
        backgroundColor: '#d69e2e',
        borderColor: '#b7791f',
        borderWidth: 1
      },
      {
        label: 'Outliers',
        data: datasets.map(dataset => qualityData.datasets[dataset].issues.outliers),
        backgroundColor: '#e53e3e',
        borderColor: '#c53030',
        borderWidth: 1
      },
      {
        label: 'Validation Errors',
        data: datasets.map(dataset => qualityData.datasets[dataset].issues.validation_errors),
        backgroundColor: '#9467bd',
        borderColor: '#805ad5',
        borderWidth: 1
      }
    ]
  };

  // Calculate metrics
  const totalIssues = datasets.reduce((sum, dataset) => {
    const issues = qualityData.datasets[dataset].issues;
    return sum + issues.missing_values + issues.outliers + issues.validation_errors;
  }, 0);

  const datasetsWithIssues = datasets.filter(dataset => {
    const issues = qualityData.datasets[dataset].issues;
    return issues.missing_values > 0 || issues.outliers > 0 || issues.validation_errors > 0;
  }).length;

  const lowestQualityDataset = datasets.reduce((lowest, dataset) => 
    qualityData.datasets[dataset].quality_score < qualityData.datasets[lowest].quality_score ? 
    dataset : lowest
  );

  return (
    <PageContainer>
      <PageTitle>Data Quality Monitoring</PageTitle>

      <MetricsGrid>
        <MetricCard
          title="Datasets Monitored"
          value={qualityData.datasets_monitored}
          loading={loading}
        />
        <MetricCard
          title="Average Quality Score"
          value={`${qualityData.average_quality_score.toFixed(1)}%`}
          status={qualityData.average_quality_score >= 90 ? 'HEALTHY' : 
                  qualityData.average_quality_score >= 70 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Total Issues"
          value={totalIssues}
          status={totalIssues === 0 ? 'HEALTHY' : totalIssues <= 10 ? 'WARNING' : 'CRITICAL'}
          loading={loading}
        />
        <MetricCard
          title="Datasets with Issues"
          value={datasetsWithIssues}
          status={datasetsWithIssues === 0 ? 'HEALTHY' : 'WARNING'}
          loading={loading}
        />
      </MetricsGrid>

      <ChartsGrid>
        <Card>
          <BarChart
            data={qualityScoreData}
            title="Quality Scores by Dataset"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <BarChart
            data={completenessData}
            title="Data Completeness by Dataset"
            height={300}
            loading={loading}
          />
        </Card>

        <Card>
          <BarChart
            data={issuesData}
            title="Issues by Dataset"
            height={300}
            stacked={true}
            loading={loading}
          />
        </Card>
      </ChartsGrid>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        {datasets.map(dataset => {
          const datasetInfo = qualityData.datasets[dataset];
          return (
            <DatasetCard key={dataset} qualityScore={datasetInfo.quality_score}>
              <DatasetName>{dataset.replace(/_/g, ' ')}</DatasetName>
              <QualityScore score={datasetInfo.quality_score}>
                {datasetInfo.quality_score.toFixed(1)}%
              </QualityScore>
              <div style={{ fontSize: '0.875rem', color: '#718096', marginBottom: '10px' }}>
                {datasetInfo.total_records.toLocaleString()} total records
              </div>
              <div style={{ fontSize: '0.875rem', color: '#718096', marginBottom: '10px' }}>
                {datasetInfo.completeness.toFixed(1)}% complete
              </div>
              
              <IssuesList>
                <IssueItem>
                  <span>Missing Values:</span>
                  <IssueCount count={datasetInfo.issues.missing_values}>
                    {datasetInfo.issues.missing_values}
                  </IssueCount>
                </IssueItem>
                <IssueItem>
                  <span>Outliers:</span>
                  <IssueCount count={datasetInfo.issues.outliers}>
                    {datasetInfo.issues.outliers}
                  </IssueCount>
                </IssueItem>
                <IssueItem>
                  <span>Validation Errors:</span>
                  <IssueCount count={datasetInfo.issues.validation_errors}>
                    {datasetInfo.issues.validation_errors}
                  </IssueCount>
                </IssueItem>
              </IssuesList>
            </DatasetCard>
          );
        })}
      </div>
    </PageContainer>
  );
};

export default DataQuality;