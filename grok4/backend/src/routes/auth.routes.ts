import { Router } from 'express';
import { body } from 'express-validator';
import { register, login, refreshToken, logout } from '../controllers/auth.controller';
import { validateRequest } from '../middleware/validateRequest';

const router = Router();

// Register
router.post(
  '/register',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters'),
    body('name').notEmpty().trim().escape()
  ],
  validateRequest,
  register
);

// Login
router.post(
  '/login',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').notEmpty()
  ],
  validateRequest,
  login
);

// Refresh token
router.post('/refresh', refreshToken);

// Logout
router.post('/logout', logout);

export default router;