"""
Grok4 Command Line Interface

Main CLI entry point for Grok4 operations.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from .utils.logger import setup_logging, get_logger
from .utils.config import Config
from .models.grok4_model import Grok4Model
from .training.trainer import Trainer

logger = get_logger(__name__)


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def cli(ctx, log_level, config):
    """Grok4 - Advanced AI Framework"""
    setup_logging(level=log_level)
    
    # Load configuration
    if config:
        ctx.obj = {'config': Config.from_file(config)}
    else:
        ctx.obj = {'config': Config()}
    
    logger.info("Grok4 CLI initialized")


@cli.command()
@click.option('--model-config', type=click.Path(exists=True), help='Model configuration file')
@click.option('--output-dir', type=click.Path(), required=True, help='Output directory for model')
@click.option('--vocab-size', type=int, default=50432, help='Vocabulary size')
@click.option('--n-layers', type=int, default=24, help='Number of layers')
@click.option('--n-heads', type=int, default=16, help='Number of attention heads')
@click.option('--d-model', type=int, default=2048, help='Model dimension')
@click.pass_context
def init_model(ctx, model_config, output_dir, vocab_size, n_layers, n_heads, d_model):
    """Initialize a new Grok4 model"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if model_config:
            config = Config.from_file(model_config)
        else:
            config = Config({
                'model': {
                    'vocab_size': vocab_size,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                    'd_model': d_model,
                }
            })
        
        # Create model
        model = Grok4Model(config=config)
        
        # Save model
        model.save_pretrained(output_path)
        config.save(output_path / "config.yaml")
        
        logger.info(f"Model initialized and saved to {output_path}")
        logger.info(f"Model parameters: {model.num_parameters():,}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)


@cli.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Training data path')
@click.option('--model-path', type=click.Path(exists=True), help='Pretrained model path')
@click.option('--config', type=click.Path(exists=True), help='Training configuration')
@click.option('--output-dir', type=click.Path(), required=True, help='Output directory')
@click.option('--val-data', type=click.Path(exists=True), help='Validation data path')
@click.option('--wandb-project', help='Wandb project name')
@click.option('--wandb-run-name', help='Wandb run name')
@click.option('--resume-from', type=click.Path(exists=True), help='Resume training from checkpoint')
@click.pass_context
def train(ctx, data, model_path, config, output_dir, val_data, wandb_project, wandb_run_name, resume_from):
    """Train a Grok4 model"""
    try:
        # Load configuration
        if config:
            train_config = Config.from_file(config)
        else:
            train_config = ctx.obj['config']
        
        # Load or create model
        if model_path:
            model = Grok4Model.from_pretrained(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            model = Grok4Model(config=train_config)
            logger.info("Created new model")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=train_config,
            tokenizer=tokenizer,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
        
        # Resume from checkpoint if specified
        if resume_from:
            trainer.load_checkpoint(Path(resume_from))
            logger.info(f"Resumed training from {resume_from}")
        
        # Start training
        trainer.train(
            dataset_path=data,
            output_dir=output_dir,
            val_dataset_path=val_data,
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Model path')
@click.option('--prompt', required=True, help='Input prompt')
@click.option('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.option('--top-p', type=float, default=0.9, help='Top-p sampling')
@click.option('--top-k', type=int, default=40, help='Top-k sampling')
@click.option('--output-file', type=click.Path(), help='Output file for generated text')
def generate(model_path, prompt, max_tokens, temperature, top_p, top_k, output_file):
    """Generate text using a trained model"""
    try:
        # Load model
        model = Grok4Model.from_pretrained(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Generate text
        generated_text = model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tokenizer=tokenizer,
        )
        
        # Output result
        result = f"Prompt: {prompt}\n\nGenerated: {generated_text}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            logger.info(f"Generated text saved to {output_file}")
        else:
            click.echo(result)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', type=int, default=8000, help='Port to bind to')
@click.option('--model-path', type=click.Path(exists=True), help='Model path')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
def serve(host, port, model_path, config):
    """Start the Grok4 API server"""
    try:
        from .api.server import main as server_main
        import sys
        
        # Prepare arguments for server
        server_args = ['--host', host, '--port', str(port)]
        
        if model_path:
            server_args.extend(['--model-path', model_path])
        
        if config:
            server_args.extend(['--config-path', config])
        
        # Replace sys.argv to pass arguments to server
        original_argv = sys.argv
        sys.argv = ['grok4-serve'] + server_args
        
        try:
            server_main()
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Model path')
@click.option('--interactive', is_flag=True, help='Interactive chat mode')
def chat(model_path, interactive):
    """Chat with a Grok4 model"""
    try:
        # Load model
        model = Grok4Model.from_pretrained(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if interactive:
            # Interactive chat loop
            click.echo("Grok4 Chat Mode (type 'quit' to exit)")
            messages = []
            
            while True:
                try:
                    user_input = click.prompt("You")
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    messages.append({"role": "user", "content": user_input})
                    
                    response = model.chat(messages=messages, tokenizer=tokenizer)
                    messages.append({"role": "assistant", "content": response})
                    
                    click.echo(f"Grok4: {response}")
                    
                except KeyboardInterrupt:
                    break
        else:
            # Single message chat
            user_input = click.prompt("Enter your message")
            messages = [{"role": "user", "content": user_input}]
            
            response = model.chat(messages=messages, tokenizer=tokenizer)
            click.echo(f"Grok4: {response}")
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Model path')
def info(model_path):
    """Display model information"""
    try:
        # Load model
        model = Grok4Model.from_pretrained(model_path)
        
        # Display information
        click.echo(f"Model Path: {model_path}")
        click.echo(f"Parameters: {model.num_parameters():,}")
        click.echo(f"Vocabulary Size: {model.vocab_size}")
        click.echo(f"Layers: {model.n_layers}")
        click.echo(f"Heads: {model.n_heads}")
        click.echo(f"Model Dimension: {model.d_model}")
        click.echo(f"Max Sequence Length: {model.max_seq_len}")
        
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    try:
        cli()
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()