import { Router } from 'express';
import { body } from 'express-validator';
import { 
  getProfile,
  updateProfile,
  deleteAccount,
  getAllUsers
} from '../controllers/user.controller';
import { validateRequest } from '../middleware/validateRequest';
import { authorize } from '../middleware/auth';

const router = Router();

// Get current user profile
router.get('/profile', getProfile);

// Update profile
router.put(
  '/profile',
  [
    body('name').optional().trim().escape(),
    body('email').optional().isEmail().normalizeEmail()
  ],
  validateRequest,
  updateProfile
);

// Delete account
router.delete('/profile', deleteAccount);

// Admin only: Get all users
router.get('/', authorize('admin'), getAllUsers);

export default router;