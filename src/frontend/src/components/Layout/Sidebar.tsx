import React from 'react';
import { NavLink } from 'react-router-dom';
import styled from 'styled-components';
import { NavItem } from '../../types';

const SidebarContainer = styled.nav`
  width: 250px;
  background: #2d3748;
  color: white;
  padding: 20px 0;
  overflow-y: auto;
`;

const Logo = styled.div`
  padding: 0 20px 30px;
  border-bottom: 1px solid #4a5568;
  margin-bottom: 20px;
`;

const LogoText = styled.h2`
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  color: #e2e8f0;
`;

const LogoSubtext = styled.p`
  font-size: 0.75rem;
  color: #a0aec0;
  margin: 4px 0 0;
`;

const NavList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
`;

const NavItemContainer = styled.li`
  margin-bottom: 4px;
`;

const NavLinkStyled = styled(NavLink)`
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #e2e8f0;
  text-decoration: none;
  transition: all 0.2s;
  border-left: 3px solid transparent;

  &:hover {
    background: #4a5568;
    color: white;
  }

  &.active {
    background: #3182ce;
    border-left-color: #63b3ed;
    color: white;
  }
`;

const NavIcon = styled.span`
  margin-right: 12px;
  font-size: 1.125rem;
  width: 20px;
  text-align: center;
`;

const NavLabel = styled.span`
  font-size: 0.875rem;
  font-weight: 500;
`;

const NavBadge = styled.span`
  background: #e53e3e;
  color: white;
  font-size: 0.75rem;
  padding: 2px 6px;
  border-radius: 10px;
  margin-left: auto;
  min-width: 18px;
  text-align: center;
`;

const Sidebar: React.FC = () => {
  const navItems: NavItem[] = [
    {
      path: '/dashboard',
      label: 'S&P 500 Analysis',
      icon: '📊'
    },
    {
      path: '/agent-dashboard',
      label: 'Agent Pipeline',
      icon: '🤖'
    },
    {
      path: '/portfolio',
      label: 'Portfolio Details',
      icon: '💼'
    },
    {
      path: '/portfolio-dashboard',
      label: 'Portfolio Dashboard',
      icon: '📈'
    },
    {
      path: '/agent-portfolio',
      label: 'AI Portfolio',
      icon: '🚀'
    },
    {
      path: '/data-quality',
      label: 'Excel Data Quality',
      icon: '🎯'
    },
    {
      path: '/investment-planner',
      label: 'Investment Planner',
      icon: '💰'
    }
  ];

  return (
    <SidebarContainer>
      <Logo>
        <LogoText>S&P 500</LogoText>
        <LogoSubtext>Historical Analysis (1927-2024)</LogoSubtext>
      </Logo>
      
      <NavList>
        {navItems.map((item) => (
          <NavItemContainer key={item.path}>
            <NavLinkStyled to={item.path}>
              <NavIcon>{item.icon}</NavIcon>
              <NavLabel>{item.label}</NavLabel>
              {item.badge && item.badge > 0 && (
                <NavBadge>{item.badge}</NavBadge>
              )}
            </NavLinkStyled>
          </NavItemContainer>
        ))}
      </NavList>
    </SidebarContainer>
  );
};

export default Sidebar;