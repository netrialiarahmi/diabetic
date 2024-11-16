import React from 'react';
import { Home, FileText } from 'lucide-react';

const NavigationBar = () => {
  const [selectedIndex, setSelectedIndex] = React.useState(0);

  const navItems = [
    { label: 'Home', icon: <Home size={18} />, href: '#' },
    { label: 'Classify MRI Scan', icon: <FileText size={18} />, href: '#' }
  ];

  return (
    <div className="w-full bg-blue-600 px-6 py-2 rounded-lg shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <a 
            href={navItems[0].href}
            onClick={() => setSelectedIndex(0)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-2xl text-white hover:bg-blue-500 transition-colors ${selectedIndex === 0 ? 'bg-blue-500' : ''}`}
          >
            {navItems[0].icon}
            <span className="text-sm">{navItems[0].label}</span>
          </a>
        </div>

        <div className="flex items-center">
          <a 
            href={navItems[1].href}
            onClick={() => setSelectedIndex(1)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-2xl text-white bg-yellow-400 hover:bg-yellow-500 transition-colors`}
          >
            {navItems[1].icon}
            <span className="text-sm">{navItems[1].label}</span>
          </a>
        </div>
      </div>
    </div>
  );
};

export default NavigationBar;