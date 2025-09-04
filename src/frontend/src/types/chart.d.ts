// Type declarations for chart.js to resolve import issues
declare module 'chart.js' {
  export * from 'chart.js/dist/chart';
}

declare module 'react-chartjs-2' {
  import { ComponentType } from 'react';
  
  export interface ChartProps {
    data: any;
    options?: any;
    [key: string]: any;
  }
  
  export const Pie: ComponentType<ChartProps>;
  export const Line: ComponentType<ChartProps>;
  export const Bar: ComponentType<ChartProps>;
}