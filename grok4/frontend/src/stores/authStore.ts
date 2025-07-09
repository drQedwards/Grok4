import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import axios from 'axios'
import toast from 'react-hot-toast'

interface User {
  id: string
  email: string
  name: string
  role: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name: string) => Promise<void>
  logout: () => void
  setUser: (user: User) => void
  setToken: (token: string) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true })
        try {
          const response = await axios.post('/api/auth/login', { email, password })
          const { user, token } = response.data
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          })
          
          // Set default auth header
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
          
          toast.success('Login successful!')
        } catch (error: any) {
          set({ isLoading: false })
          toast.error(error.response?.data?.error || 'Login failed')
          throw error
        }
      },

      register: async (email: string, password: string, name: string) => {
        set({ isLoading: true })
        try {
          const response = await axios.post('/api/auth/register', {
            email,
            password,
            name,
          })
          const { user, token } = response.data
          
          set({
            user,
            token,
            isAuthenticated: true,
            isLoading: false,
          })
          
          // Set default auth header
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
          
          toast.success('Registration successful!')
        } catch (error: any) {
          set({ isLoading: false })
          toast.error(error.response?.data?.error || 'Registration failed')
          throw error
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        })
        
        // Remove auth header
        delete axios.defaults.headers.common['Authorization']
        
        toast.success('Logged out successfully')
      },

      setUser: (user: User) => set({ user }),
      setToken: (token: string) => {
        set({ token, isAuthenticated: true })
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)

// Initialize axios auth header if token exists
const token = useAuthStore.getState().token
if (token) {
  axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
}