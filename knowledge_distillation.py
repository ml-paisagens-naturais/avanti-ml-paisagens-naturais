"""
Módulo para destilação de conhecimento de modelos maiores para menores.

Este módulo implementa técnicas de destilação de conhecimento para transferir
aprendizado de modelos grandes (professores) para modelos menores (alunos).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Obter o logger
logger = logging.getLogger("landscape_classifier")

class DistillationLoss(nn.Module):
    """
    Implementa a perda de destilação de conhecimento combinando perda dura (hard) e macia (soft).
    
    Args:
        alpha: Peso da perda com rótulos reais (hard loss)
        temperature: Temperatura para suavizar as distribuições de probabilidade
    """
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_criterion = nn.CrossEntropyLoss()
        self.soft_criterion = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, targets):
        """
        Calcula perda combinada (hard e soft).
        
        Args:
            student_logits: Saídas do modelo estudante
            teacher_logits: Saídas do modelo professor
            targets: Rótulos das classes reais
            
        Returns:
            torch.Tensor: Perda total combinada
        """
        # Perda com rótulos reais (CrossEntropy)
        hard_loss = self.hard_criterion(student_logits, targets)
        
        # Perda de destilação (KL divergence)
        # Aplicar temperatura às distribuições
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL Divergence com temperatura
        soft_loss = self.soft_criterion(log_probs, soft_targets) * (self.temperature**2)
        
        # Perda combinada
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

def distill_knowledge(student_model, teacher_model, train_loader, test_loader, 
                      optimizer, scheduler=None, num_epochs=10, 
                      alpha=0.5, temperature=3.0, device=None,
                      teacher_name="MobileNet", student_name="NatureLightNet",
                      save_path="models/naturelight_distilled.pth"):
    """
    Treina o modelo estudante via destilação de conhecimento do modelo professor.
    
    Args:
        student_model: Modelo menor a ser treinado (estudante)
        teacher_model: Modelo maior que transfere conhecimento (professor)
        train_loader: DataLoader para dados de treinamento
        test_loader: DataLoader para dados de teste/validação
        optimizer: Otimizador para o modelo estudante
        scheduler: Scheduler de learning rate (opcional)
        num_epochs: Número de épocas de treinamento
        alpha: Peso da perda com rótulos reais (0-1)
        temperature: Temperatura para destilação
        device: Dispositivo para processamento (CPU/GPU)
        teacher_name: Nome do modelo professor para logging
        student_name: Nome do modelo estudante para logging
        save_path: Caminho para salvar o modelo final
    
    Returns:
        tuple: (modelo_treinado, histórico_treino, histórico_validação)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Iniciando destilação de conhecimento: {teacher_name} → {student_name}")
    logger.info(f"Alpha={alpha}, Temperatura={temperature}, Dispositivo={device}")
    
    # Mover modelos para o dispositivo
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # Configurar modos de treinamento
    student_model.train()
    teacher_model.eval()  # Professor sempre em modo de avaliação
    
    # Inicializar critério de destilação
    distillation_criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    
    # Inicializar writer TensorBoard
    writer = SummaryWriter(log_dir=f"tensorboard_logs/distillation_{student_name}")
    
    # Métricas para acompanhamento
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # Medir tempo inicial
    start_time = time.time()
    
    # Loop de épocas
    for epoch in range(num_epochs):
        logger.info(f"Época {epoch+1}/{num_epochs}")
        
        # === TREINAMENTO ===
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Forward pass do estudante
            student_logits = student_model(inputs)
            
            # Forward pass do professor (sem gradientes)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # Calcular perda de destilação
            loss = distillation_criterion(student_logits, teacher_logits, targets)
            
            # Backward e otimização
            loss.backward()
            optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log de progresso a cada 20 batches
            if batch_idx % 20 == 0:
                logger.debug(f"Treinamento: Batch {batch_idx}/{len(train_loader)}")
                
                # Adicionar gráficos ao TensorBoard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Distillation/batch_loss", loss.item(), step)
                
                # Visualizar distribuições do professor vs. estudante para o primeiro batch
                if batch_idx == 0:
                    teacher_probs = F.softmax(teacher_logits[:8], dim=1).cpu().numpy()
                    student_probs = F.softmax(student_logits[:8], dim=1).cpu().numpy()
                    
                    for i in range(min(4, len(teacher_probs))):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        x = np.arange(len(teacher_probs[i]))
                        width = 0.35
                        ax.bar(x - width/2, teacher_probs[i], width, label=f'{teacher_name}')
                        ax.bar(x + width/2, student_probs[i], width, label=f'{student_name}')
                        ax.set_xticks(x)
                        ax.set_title(f'Distribuição de Probabilidades (Imagem {i+1})')
                        ax.set_ylabel('Probabilidade')
                        ax.set_xlabel('Classe')
                        ax.legend()
                        writer.add_figure(f"Distillation/probs_epoch{epoch}_sample{i}", fig, epoch)
                        plt.close(fig)
                        
        # Calcular métricas da época
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        logger.info(f"Treinamento => Perda: {train_loss:.4f}, Acurácia: {train_acc:.2f}%")
        
        # === VALIDAÇÃO ===
        student_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass do estudante
                student_logits = student_model(inputs)
                
                # Forward pass do professor
                teacher_logits = teacher_model(inputs)
                
                # Calcular perda de destilação
                loss = distillation_criterion(student_logits, teacher_logits, targets)
                
                # Estatísticas
                running_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calcular métricas de validação
        val_loss = running_loss / len(test_loader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"Validação => Perda: {val_loss:.4f}, Acurácia: {val_acc:.2f}%")
        
        # Atualizar TensorBoard
        writer.add_scalar("Distillation/train_loss", train_loss, epoch)
        writer.add_scalar("Distillation/val_loss", val_loss, epoch)
        writer.add_scalar("Distillation/train_accuracy", train_acc, epoch)
        writer.add_scalar("Distillation/val_accuracy", val_acc, epoch)
        
        # Ajustar learning rate se houver scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
            # Registrar learning rate atual
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Distillation/learning_rate", current_lr, epoch)
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            logger.info(f"Acurácia de validação melhorou: {best_val_acc:.2f}% → {val_acc:.2f}%")
            best_val_acc = val_acc
            
            # Salvar checkpoint
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'parameters': {
                    'alpha': alpha,
                    'temperature': temperature
                }
            }, save_path)
            logger.info(f"Melhor modelo salvo em {save_path}")
    
    # Tempo total de treinamento
    total_time = time.time() - start_time
    logger.info(f"Destilação concluída em {total_time/60:.2f} minutos")
    logger.info(f"Melhor acurácia de validação: {best_val_acc:.2f}%")
    
    # Criar gráficos finais
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, num_epochs+1)
    
    # Gráfico de perda
    ax1.plot(epochs_range, train_losses, 'b-', label='Treino')
    ax1.plot(epochs_range, val_losses, 'r-', label='Validação')
    ax1.set_title('Curvas de Perda - Destilação')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de acurácia
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Treino')
    ax2.plot(epochs_range, val_accuracies, 'r-', label='Validação')
    ax2.set_title('Curvas de Acurácia - Destilação')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    writer.add_figure("Distillation/training_curves", fig, 0)
    plt.savefig(f"results/distillation_{student_name}_curves.png")
    plt.close(fig)
    
    # Fechar TensorBoard writer
    writer.close()
    
    # Carregar melhor modelo
    checkpoint = torch.load(save_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    return student_model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_accuracy': best_val_acc
    }

def train_naturelight_with_distillation(train_dataset, test_dataset, device=None, 
                                       num_epochs=10, save_path="models/naturelight_distilled.pth"):
    """
    Função simplificada para treinar o NatureLightNet usando destilação de conhecimento do MobileNet.
    
    Args:
        train_dataset: Dataset de treinamento
        test_dataset: Dataset de teste
        device: Dispositivo para treinamento (CPU/GPU)
        num_epochs: Número de épocas para treinamento
        save_path: Caminho para salvar o modelo treinado
        
    Returns:
        tuple: (modelo_treinado, histórico_treino)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info(f"Preparando destilação MobileNet → NatureLightNet em {device}")
    
    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Criar modelo estudante (NatureLightNet)
    from NatureLightNet import create_naturelight_model
    student_model = create_naturelight_model(num_classes=6)
    student_model = student_model.to(device)
    
    # Criar modelo professor (MobileNet)
    import torchvision.models as tvm
    teacher_model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Verificar se o modelo pré-treinado existe, caso contrário, usar padrão
    pretrained_path = "models/mobilenet_final.pth"
    if os.path.exists(pretrained_path):
        try:
            # Adaptar último classificador para 6 classes
            num_classes = 6
            in_features = teacher_model.classifier[-1].in_features
            teacher_model.classifier[-1] = nn.Linear(in_features, num_classes)
            
            # Carregar pesos pré-treinados
            teacher_model.load_state_dict(torch.load(pretrained_path))
            logger.info("Modelo MobileNet pré-treinado carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo pré-treinado: {str(e)}")
            logger.info("Usando modelo MobileNet padrão (alterar último classificador)")
            
            # Adaptar último classificador para 6 classes
            in_features = teacher_model.classifier[-1].in_features
            teacher_model.classifier[-1] = nn.Linear(in_features, 6)
    else:
        logger.info("Modelo MobileNet pré-treinado não encontrado, usando modelo padrão")
        
        # Adaptar último classificador para 6 classes
        in_features = teacher_model.classifier[-1].in_features
        teacher_model.classifier[-1] = nn.Linear(in_features, 6)
    
    teacher_model = teacher_model.to(device)
    
    # Configurar otimizador e scheduler para o estudante
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Iniciar destilação
    return distill_knowledge(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        alpha=0.5,  # Balanceamento entre hard e soft loss
        temperature=3.0,  # Temperatura para suavizar as distribuições
        device=device,
        save_path=save_path
    )