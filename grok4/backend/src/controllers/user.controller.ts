import { Request, Response } from 'express';
import { logger } from '../utils/logger';

interface AuthRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

// Mock data - in real app, this would come from database
const users = new Map();

export const getProfile = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const user = Array.from(users.values()).find(u => u.id === req.user?.id);
    
    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    logger.error('Get profile error:', error);
    res.status(500).json({ error: 'Failed to get profile' });
  }
};

export const updateProfile = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const { name, email } = req.body;
    const user = Array.from(users.values()).find(u => u.id === req.user?.id);
    
    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    // Update user
    if (name) user.name = name;
    if (email && email !== user.email) {
      // Check if email is taken
      if (users.has(email)) {
        res.status(400).json({ error: 'Email already in use' });
        return;
      }
      users.delete(user.email);
      user.email = email;
      users.set(email, user);
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role
      }
    });
  } catch (error) {
    logger.error('Update profile error:', error);
    res.status(500).json({ error: 'Failed to update profile' });
  }
};

export const deleteAccount = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const user = Array.from(users.values()).find(u => u.id === req.user?.id);
    
    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    users.delete(user.email);
    res.json({ success: true, message: 'Account deleted successfully' });
  } catch (error) {
    logger.error('Delete account error:', error);
    res.status(500).json({ error: 'Failed to delete account' });
  }
};

export const getAllUsers = async (req: AuthRequest, res: Response): Promise<void> => {
  try {
    const userList = Array.from(users.values()).map(({ password, ...user }) => user);
    
    res.json({
      success: true,
      users: userList,
      total: userList.length
    });
  } catch (error) {
    logger.error('Get all users error:', error);
    res.status(500).json({ error: 'Failed to get users' });
  }
};