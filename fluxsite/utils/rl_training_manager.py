from fluxsite.utils.training_utils_enhanced import EnhancedTrainingManager
from fluxsite.utils.rl_controller import RLHyperparameterController
import torch
import os

class RLEnhancedTrainingManager(EnhancedTrainingManager):
    """
    Extended Training Manager with Reinforcement Learning Controller.
    """
    def __init__(self, model, config, train_loader, val_loader, optimizer, scheduler, device, output_dir):
        super().__init__(model, config, train_loader, val_loader, optimizer, scheduler, device, output_dir)
        
        # Initialize RL Controller
        self.rl_controller = RLHyperparameterController(
            training_manager=self,
            config=self.config,
            output_dir=self.output_dir
        )
        
    def train(self):
        """
        Override train method to include RL step.
        """
        self.logger.info("=== Start RL Enhanced Training ===")
        
        # ... (Copying essential parts of the original train loop structure) ...
        # Since we cannot easily inject into the middle of the original train() method without copying it entirely,
        # we will copy the loop logic here.
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_mcc = 0.0
        best_val_auprc = 0.0
        best_epoch = 0
        
        num_epochs = self.config.get('epochs', 50)
        
        for epoch in range(num_epochs):
            self.logger.info("="*80)
            self.logger.info(f"StartTraining Epoch {epoch+1}/{num_epochs} (RL Enabled)")
            self.logger.info("="*80)
            
            self._update_training_stage(epoch)

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate_epoch(epoch)

            # --- RL Step ---
            self.rl_controller.step(train_metrics, val_metrics, epoch)
            # ---------------

            # Update adaptive dropout (Original logic, might conflict with RL, but we keep it as base)
            if self.dropout_scheduler is not None:
                self.dropout_scheduler.step(epoch, val_metrics['val_loss'])

            if hasattr(self.model, 'update_dropout_schedule'):
                self.model.update_dropout_schedule(epoch, val_metrics['val_loss'])

            self._update_history(train_metrics, val_metrics)
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)
            self._step_scheduler_epoch(epoch, val_metrics)
            
            should_stop, improvement_flag = self._check_early_stopping(
                val_metrics, epoch, best_val_loss, best_val_acc, best_val_f1,
                best_val_mcc, best_val_auprc, best_epoch
            )

            if improvement_flag.find(" BestModel") != -1:
                best_val_loss = val_metrics['val_loss']
                best_val_acc = val_metrics['val_acc']
                best_val_f1 = val_metrics.get('val_f1', 0.0)
                best_val_mcc = val_metrics.get('val_mcc', 0.0)
                best_val_auprc = val_metrics.get('val_auprc', 0.0)
                best_epoch = epoch

            if should_stop:
                self.logger.info("="*80)
                self.logger.info(f", {epoch+1} Training")
                break
            
            self.logger.info("="*60)
            self.logger.info(f"Epoch {epoch+1} {improvement_flag}")
            self.logger.info(f"Training: Loss={train_metrics.get('loss', 0):.4f}, Acc={train_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"Validation: Loss={val_metrics['val_loss']:.4f}, Acc={val_metrics['val_acc']:.4f}")
            self.logger.info("="*60)
        
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'history': self.training_history
        }
