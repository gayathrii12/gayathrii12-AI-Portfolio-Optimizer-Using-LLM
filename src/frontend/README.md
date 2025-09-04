# Financial Returns Optimizer - React Frontend

A comprehensive React.js dashboard for monitoring and visualizing the Financial Returns Optimizer system with real-time charts, performance metrics, and system health monitoring.

## Features

### 📊 **Interactive Dashboard**
- Real-time system status monitoring
- Component activity visualization
- Performance metrics overview
- Error and warning tracking

### ⚡ **Performance Monitoring**
- Component performance analysis
- Success rate tracking
- Duration monitoring
- Performance trends over time

### 🎯 **Data Quality Monitoring**
- Dataset quality scores
- Data completeness tracking
- Issue detection and reporting
- Quality trends visualization

### 🚨 **Error Tracking**
- Real-time error monitoring
- Error categorization by type and component
- Recent error timeline
- Error rate analysis

### 💼 **Portfolio Analysis**
- Interactive pie charts for allocation
- Performance line charts
- Portfolio vs benchmark comparison
- Risk analysis and metrics

### ❤️ **System Health**
- Overall system status
- Health metrics dashboard
- Issue identification
- Automated recommendations

## Technology Stack

- **React 18** with TypeScript
- **Chart.js** and **React-Chart.js-2** for interactive charts
- **Styled Components** for styling
- **Axios** for API communication
- **React Router** for navigation

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Python backend server running on port 8000

### Installation

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm start
   ```

3. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Backend API

The frontend expects a backend API server running on `http://localhost:8000`. To start the backend:

```bash
# From the project root directory
python backend_api.py
```

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── Charts/
│   │   │   ├── PieChart.tsx
│   │   │   ├── LineChart.tsx
│   │   │   └── BarChart.tsx
│   │   ├── Common/
│   │   │   ├── StatusBadge.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   └── Layout/
│   │       ├── Header.tsx
│   │       └── Sidebar.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── PerformanceMonitoring.tsx
│   │   ├── DataQuality.tsx
│   │   ├── ErrorTracking.tsx
│   │   ├── PortfolioAnalysis.tsx
│   │   └── SystemHealth.tsx
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── index.tsx
│   └── index.css
├── package.json
└── tsconfig.json
```

## API Integration

The frontend communicates with the backend through REST API endpoints:

- `GET /api/dashboard` - Dashboard overview data
- `GET /api/system-health` - System health status
- `GET /api/performance/summary` - Performance metrics
- `GET /api/data-quality/summary` - Data quality metrics
- `GET /api/errors/summary` - Error tracking data
- `GET /api/portfolio/*` - Portfolio analysis data

## Features Overview

### Dashboard Page
- System status indicator
- Key performance metrics
- Component activity charts
- System health summary
- Automated recommendations

### Performance Monitoring
- Component performance overview
- Success rate tracking
- Performance trends
- Detailed performance tables

### Data Quality
- Quality score visualization
- Data completeness metrics
- Issue tracking by dataset
- Quality trend analysis

### Error Tracking
- Real-time error monitoring
- Error categorization
- Recent error timeline
- Component error analysis

### Portfolio Analysis
- Asset allocation pie charts
- Performance line charts
- Benchmark comparison
- Risk analysis dashboard

### System Health
- Overall system status
- Performance indicators
- Issue identification
- Health recommendations

## Customization

### Adding New Charts
1. Create a new chart component in `src/components/Charts/`
2. Import and use Chart.js or Recharts
3. Add TypeScript interfaces in `src/types/index.ts`
4. Integrate with API service in `src/services/api.ts`

### Adding New Pages
1. Create a new page component in `src/pages/`
2. Add route in `src/App.tsx`
3. Update navigation in `src/components/Layout/Sidebar.tsx`

### Styling
The app uses Styled Components for styling. Global styles are in `src/index.css`.

## Environment Variables

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:8000
```

## Production Build

To create a production build:

```bash
npm run build
```

This creates a `build` folder with optimized production files.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Follow TypeScript best practices
2. Use Styled Components for styling
3. Add proper error handling
4. Include loading states
5. Write meaningful component names
6. Add proper TypeScript interfaces

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure backend server is running on port 8000
   - Check CORS configuration
   - Verify API endpoints

2. **Charts Not Rendering**
   - Check Chart.js dependencies
   - Verify data format matches chart requirements
   - Check console for JavaScript errors

3. **TypeScript Errors**
   - Ensure all interfaces are properly defined
   - Check import statements
   - Verify prop types match interfaces

### Performance Tips

1. Use React.memo for expensive components
2. Implement proper loading states
3. Use lazy loading for large datasets
4. Optimize chart rendering with proper keys
5. Implement proper error boundaries