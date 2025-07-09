import { Router } from 'express';
import authRoutes from './auth.routes';
import userRoutes from './user.routes';
import { authenticate } from '../middleware/auth';

const router = Router();

// Public routes
router.use('/auth', authRoutes);

// Protected routes
router.use('/users', authenticate, userRoutes);

// Welcome route
router.get('/', (req, res) => {
  res.json({
    message: 'Welcome to Grok4 API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      users: '/api/users',
      health: '/health'
    }
  });
});

export default router;