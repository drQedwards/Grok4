# Grok4 - Modern Full-Stack Application

A production-ready full-stack application built with React, TypeScript, Node.js, and Express. Features include authentication, user management, and a beautiful modern UI.

## 🚀 Features

- **Frontend**
  - React 18 with TypeScript
  - Vite for blazing fast development
  - Tailwind CSS for styling
  - React Router for navigation
  - Zustand for state management
  - React Query for data fetching
  - React Hook Form for forms
  - Lucide React for icons

- **Backend**
  - Node.js with Express
  - TypeScript for type safety
  - JWT authentication
  - MongoDB with Mongoose
  - Redis for caching (optional)
  - Rate limiting and security headers
  - Input validation with express-validator

- **DevOps**
  - Docker and Docker Compose support
  - Environment-based configuration
  - Health check endpoints
  - Production-ready logging

## 📋 Prerequisites

- Node.js 18+ and npm
- MongoDB (local or cloud instance)
- Redis (optional, for caching)
- Docker and Docker Compose (optional)

## 🛠️ Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/grok4.git
cd grok4
```

### Install dependencies
```bash
# Install root dependencies
npm install

# Install all dependencies (frontend and backend)
npm run install:all
```

### Environment Setup
```bash
# Copy the example environment file
cp backend/.env.example backend/.env

# Edit the .env file with your configuration
# Important: Change JWT_SECRET to a secure random string
```

## 🚀 Development

### Start the development servers

```bash
# Start both frontend and backend in development mode
npm run dev

# Or start them separately:
# Backend only (http://localhost:5000)
cd backend && npm run dev

# Frontend only (http://localhost:3000)
cd frontend && npm run dev
```

### Access the application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- API Documentation: http://localhost:5000/api

## 🏗️ Building for Production

### Build the applications
```bash
# Build both frontend and backend
npm run build

# Or build separately:
cd backend && npm run build
cd frontend && npm run build
```

### Run in production mode
```bash
# Using Node.js
cd backend && npm start

# Using Docker Compose (recommended)
docker-compose up -d
```

## 🐳 Docker Deployment

### Build and run with Docker Compose
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Development with Docker
```bash
# Start with development profile (includes hot reload)
docker-compose --profile dev up
```

## 📁 Project Structure

```
grok4/
├── backend/              # Backend Node.js application
│   ├── src/
│   │   ├── config/      # Configuration files
│   │   ├── controllers/ # Route controllers
│   │   ├── middleware/  # Express middleware
│   │   ├── models/      # Database models
│   │   ├── routes/      # API routes
│   │   ├── services/    # Business logic
│   │   ├── utils/       # Utility functions
│   │   └── server.ts    # Server entry point
│   ├── tests/           # Backend tests
│   └── package.json
├── frontend/            # Frontend React application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API services
│   │   ├── stores/      # State management
│   │   ├── hooks/       # Custom React hooks
│   │   ├── types/       # TypeScript types
│   │   ├── utils/       # Utility functions
│   │   ├── App.tsx      # Main App component
│   │   └── main.tsx     # Application entry point
│   ├── public/          # Static assets
│   └── package.json
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile          # Docker build configuration
└── README.md          # This file
```

## 🔧 Configuration

### Backend Environment Variables
```env
# Server
PORT=5000
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/grok4

# Authentication
JWT_SECRET=your-super-secret-jwt-key

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000

# Redis (optional)
REDIS_URL=redis://localhost:6379
```

### Frontend Environment Variables
```env
# API URL
VITE_API_URL=http://localhost:5000
```

## 📚 API Documentation

### Authentication Endpoints
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/refresh` - Refresh JWT token
- `POST /api/auth/logout` - Logout user

### User Endpoints (Protected)
- `GET /api/users/profile` - Get current user profile
- `PUT /api/users/profile` - Update user profile
- `DELETE /api/users/profile` - Delete user account
- `GET /api/users` - Get all users (admin only)

### Health Check
- `GET /health` - Application health status

## 🧪 Testing

```bash
# Run backend tests
cd backend && npm test

# Run frontend tests
cd frontend && npm test

# Run all tests
npm test
```

## 🔒 Security

- JWT authentication with secure token storage
- Password hashing with bcrypt
- Input validation and sanitization
- Rate limiting on API endpoints
- CORS configuration
- Security headers with Helmet
- Environment variable protection

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 💡 Troubleshooting

### Common Issues

**MongoDB Connection Error**
- Ensure MongoDB is running locally or update `MONGODB_URI` in `.env`

**Port Already in Use**
- Change the port in `.env` file or kill the process using the port

**Dependencies Installation Issues**
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

## 📞 Support

For support, email support@grok4.dev or open an issue on GitHub.

---

Built with ❤️ by the Grok4 Team