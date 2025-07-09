import { useAuthStore } from '../stores/authStore'
import { Activity, Users, FileText, TrendingUp } from 'lucide-react'

export default function DashboardPage() {
  const { user } = useAuthStore()

  const stats = [
    { name: 'Total Users', value: '1,234', icon: Users, change: '+12%' },
    { name: 'Active Sessions', value: '456', icon: Activity, change: '+8%' },
    { name: 'Documents', value: '789', icon: FileText, change: '+23%' },
    { name: 'Growth Rate', value: '15.3%', icon: TrendingUp, change: '+4.2%' },
  ]

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome back, {user?.name}!
        </h1>
        <p className="text-gray-600 mt-2">
          Here's what's happening with your application today.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => (
          <div key={stat.name} className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{stat.name}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                <p className="text-sm text-green-600 mt-1">{stat.change}</p>
              </div>
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <stat.icon className="w-6 h-6 text-primary-600" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-primary-600 rounded-full mt-2"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900">
                    User action performed on resource #{i}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {i} hours ago
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
          <div className="space-y-3">
            <button className="w-full btn-primary text-left">
              Create New Project
            </button>
            <button className="w-full btn-secondary text-left">
              Invite Team Member
            </button>
            <button className="w-full btn-secondary text-left">
              Generate Report
            </button>
            <button className="w-full btn-secondary text-left">
              View Analytics
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}