"""
Módulo para técnicas avançadas de amostragem e redução de dados.

Este módulo implementa estratégias para seleção inteligente de subconjuntos
de dados para treinamento, incluindo amostragem baseada em diversidade (coreset)
e amostragem adaptativa baseada em dificuldade.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

# Obter o logger
logger = logging.getLogger("landscape_classifier")

def extract_features(model, dataset, num_samples=None, batch_size=64, device=None):
    """
    Extrai características de um dataset usando um modelo pré-treinado.
    
    Args:
        model: Modelo PyTorch para extração de características
        dataset: Dataset de entrada
        num_samples: Número de amostras para extrair (None = todas)
        batch_size: Tamanho do batch para inferência
        device: Dispositivo para processamento
        
    Returns:
        tuple: (features, indices)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configurar para inferência
    model.eval()
    model = model.to(device)
    
    # Limitar número de amostras
    if num_samples is None or num_samples >= len(dataset):
        indices = range(len(dataset))
    else:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features_list = []
    
    logger.info(f"Extraindo características de {len(indices)} imagens...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            
            # Para ResNet, EfficientNet e modelos similares
            if hasattr(model, 'features') and callable(getattr(model, 'features', None)):
                # Para modelos com método .features()
                features = model.features(inputs)
            elif hasattr(model, 'forward_features') and callable(getattr(model, 'forward_features', None)):
                # Para Vision Transformers
                features = model.forward_features(inputs)
            else:
                # Fallback: remover a última camada
                # Executar até a penúltima camada
                # Isso é uma simplificação, pode não funcionar para todos os modelos
                x = inputs
                for name, module in model.named_children():
                    if name == 'fc' or name == 'classifier' or name == 'head':
                        break
                    x = module(x)
                features = x
            
            # Global average pooling para características 2D
            if len(features.shape) > 2:
                features = torch.mean(features, dim=[2, 3])
            
            features_list.append(features.cpu())
    
    # Concatenar todas as características
    features = torch.cat(features_list).numpy()
    
    logger.info(f"Características extraídas com forma: {features.shape}")
    return features, indices

def select_diverse_coreset(dataset, num_samples, feature_extractor=None, device=None):
    """
    Seleciona um subconjunto representativo que maximiza a diversidade.
    
    Args:
        dataset: Dataset completo
        num_samples: Número de amostras a selecionar
        feature_extractor: Modelo para extração de características (None = usar MobileNet)
        device: Dispositivo para processamento
        
    Returns:
        list: Índices das amostras selecionadas
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Selecionando coreset diverso com {num_samples} amostras...")
    
    # Se não tiver um extrator, usar MobileNet
    if feature_extractor is None:
        import torchvision.models as models
        feature_extractor = models.mobilenet_v3_small(weights='DEFAULT')
        feature_extractor = feature_extractor.to(device)
        logger.info("Usando MobileNet como extrator de características")
    
    # Extrair características
    features, indices = extract_features(feature_extractor, dataset, device=device)
    
    # Iniciar com uma amostra aleatória como primeiro ponto
    selected_indices = [np.random.randint(0, len(indices))]
    selected_features = features[selected_indices]
    remaining_indices = np.setdiff1d(np.arange(len(features)), selected_indices)
    
    logger.info(f"Iniciando seleção greedy com {num_samples} amostras...")
    
    # Selecionar amostras progressivamente - algoritmo greedy
    pbar = tqdm(total=num_samples-1)
    while len(selected_indices) < num_samples and len(remaining_indices) > 0:
        # Calcular distâncias entre amostras restantes e conjunto já selecionado
        # Usamos a distância ao ponto mais próximo
        min_distances = pairwise_distances(
            features[remaining_indices], selected_features, metric='euclidean'
        ).min(axis=1)
        
        # Selecionar a amostra com maior distância mínima (mais distante do conjunto)
        furthest_idx = remaining_indices[min_distances.argmax()]
        
        # Adicionar ao conjunto selecionado
        selected_indices.append(furthest_idx)
        selected_features = np.vstack([selected_features, features[furthest_idx:furthest_idx+1]])
        
        # Atualizar índices restantes
        remaining_indices = np.setdiff1d(remaining_indices, [furthest_idx])
        pbar.update(1)
    
    pbar.close()
    
    # Converter para índices originais
    selected_original_indices = [indices[i] for i in selected_indices]
    
    logger.info(f"Coreset selecionado com {len(selected_original_indices)} amostras")
    return selected_original_indices

def select_coreset_kmeans(dataset, num_samples, feature_extractor=None, device=None):
    """
    Seleciona um subconjunto representativo usando K-means clustering.
    
    Args:
        dataset: Dataset completo
        num_samples: Número de amostras a selecionar
        feature_extractor: Modelo para extração de características (None = usar MobileNet)
        device: Dispositivo para processamento
        
    Returns:
        list: Índices das amostras selecionadas
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Selecionando coreset usando K-means com {num_samples} clusters...")
    
    # Se não tiver um extrator, usar MobileNet
    if feature_extractor is None:
        import torchvision.models as models
        feature_extractor = models.mobilenet_v3_small(weights='DEFAULT')
        feature_extractor = feature_extractor.to(device)
        logger.info("Usando MobileNet como extrator de características")
    
    # Extrair características
    features, indices = extract_features(feature_extractor, dataset, device=device)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Selecionar uma amostra de cada cluster (a mais próxima do centróide)
    selected_indices = []
    
    for i in range(num_samples):
        cluster_samples = np.where(cluster_labels == i)[0]
        if len(cluster_samples) > 0:
            # Calcular distâncias ao centróide do cluster
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(features[cluster_samples] - centroid, axis=1)
            
            # Selecionar amostra mais próxima do centróide
            closest_idx = cluster_samples[distances.argmin()]
            selected_indices.append(indices[closest_idx])
    
    logger.info(f"Coreset K-means selecionado com {len(selected_indices)} amostras")
    return selected_indices

class DifficultyDataset(Dataset):
    """
    Dataset wrapper que mantém informações sobre dificuldade das amostras
    """
    def __init__(self, base_dataset, difficulty_scores=None):
        self.base_dataset = base_dataset
        
        # Inicializar scores de dificuldade com zeros
        if difficulty_scores is None:
            self.difficulty_scores = np.zeros(len(base_dataset))
        else:
            self.difficulty_scores = difficulty_scores
    
    def __getitem__(self, idx):
        sample, target = self.base_dataset[idx]
        return sample, target, self.difficulty_scores[idx]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def update_difficulty(self, indices, new_scores):
        """Atualiza os scores de dificuldade para os índices fornecidos"""
        self.difficulty_scores[indices] = new_scores

def difficulty_based_sampling(model, dataset, initial_ratio=0.3, final_ratio=0.7, 
                             hard_fraction=0.7, epochs=10, device=None):
    """
    Implementa amostragem adaptativa baseada na dificuldade das amostras.
    
    Args:
        model: Modelo para avaliar dificuldade
        dataset: Dataset completo
        initial_ratio: Fração inicial de dados para usar
        final_ratio: Fração final de dados para usar
        hard_fraction: Fração de amostras difíceis a selecionar
        epochs: Número de épocas para adaptação
        device: Dispositivo para processamento
        
    Returns:
        list: Listas de índices selecionados para cada época
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Total de amostras
    num_samples = len(dataset)
    logger.info(f"Iniciando amostragem baseada em dificuldade: {initial_ratio*100:.1f}% -> {final_ratio*100:.1f}%")
    
    # Lista para armazenar índices selecionados em cada época
    epoch_indices = []
    
    # Iniciar com subset aleatório
    initial_size = int(num_samples * initial_ratio)
    current_indices = np.random.choice(
        np.arange(num_samples), initial_size, replace=False
    )
    epoch_indices.append(current_indices.copy())
    
    # Criar dataloader para todo o dataset
    batch_size = 32
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Loop principal de épocas
    for epoch in range(1, epochs):
        logger.info(f"Época {epoch}/{epochs-1} - Avaliando dificuldade das amostras")
        
        # Avaliar dificuldade de todas as amostras
        difficulty_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(full_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                # Calcular probabilidade da classe correta
                # Menor probabilidade = exemplo mais difícil
                correct_probs = probs[range(len(targets)), targets]
                
                # Converter para CPU e NumPy
                difficulty_scores.extend(correct_probs.cpu().numpy())
        
        # Normalizar scores (0 = mais difícil, 1 = mais fácil)
        difficulty_scores = np.array(difficulty_scores)
        
        # Calcular tamanho do subset nesta época (crescente)
        ratio = initial_ratio + (final_ratio - initial_ratio) * (epoch / (epochs - 1))
        new_size = int(num_samples * ratio)
        
        # Número de amostras difíceis e aleatórias
        num_hard = int(new_size * hard_fraction)
        num_random = new_size - num_hard
        
        # Selecionar exemplos mais difíceis
        hard_indices = np.argsort(difficulty_scores)[:num_hard]
        
        # Selecionar exemplos aleatórios dos restantes
        remaining_indices = np.setdiff1d(np.arange(num_samples), hard_indices)
        random_indices = np.random.choice(remaining_indices, num_random, replace=False)
        
        # Combinar índices
        current_indices = np.concatenate([hard_indices, random_indices])
        epoch_indices.append(current_indices.copy())
        
        logger.info(f"Época {epoch}: selecionadas {len(current_indices)} amostras")
        logger.info(f"  - {num_hard} amostras difíceis, {num_random} amostras aleatórias")
        logger.info(f"  - Média de dificuldade (subset): {difficulty_scores[current_indices].mean():.4f}")
        logger.info(f"  - Média de dificuldade (total): {difficulty_scores.mean():.4f}")
    
    return epoch_indices

def train_with_difficulty_sampling(model, dataset, test_dataset, optimizer, criterion,
                                  initial_ratio=0.3, final_ratio=0.7, hard_fraction=0.7,
                                  epochs=10, batch_size=32, scheduler=None, 
                                  writer=None, device=None):
    """
    Treina modelo com amostragem adaptativa baseada em dificuldade.
    
    Args:
        model: Modelo a ser treinado
        dataset: Dataset completo
        test_dataset: Dataset de teste
        optimizer: Otimizador configurado
        criterion: Função de perda
        initial_ratio: Fração inicial de dados (ex: 0.3 = 30%)
        final_ratio: Fração final de dados (ex: 0.7 = 70%)
        hard_fraction: Fração de amostras difíceis (vs aleatórias)
        epochs: Número de épocas
        batch_size: Tamanho do batch
        scheduler: Scheduler de learning rate (opcional)
        writer: TensorBoard writer (opcional)
        device: Dispositivo para treinamento
        
    Returns:
        tuple: (modelo_treinado, histórico_treino)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Dataloader de teste
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Métricas para acompanhamento
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    difficulty_evolution = []
    subset_sizes = []
    best_acc = 0.0
    
    # Configurar TensorBoard writer se não fornecido
    if writer is None:
        writer = SummaryWriter(log_dir=f"tensorboard_logs/difficulty_sampling")
    
    # Iniciar timer
    start_time = time.time()
    
    # Loop principal de épocas
    for epoch in range(epochs):
        logger.info(f"Época {epoch+1}/{epochs}")
        
        # === FASE 1: AVALIAÇÃO DE DIFICULDADE ===
        if epoch > 0:  # Pular na primeira época
            logger.info("Avaliando dificuldade das amostras...")
            model.eval()
            
            difficulty_scores = []
            all_indices = []
            
            # Dataloader para todo o dataset
            full_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(full_loader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Calcular índices originais
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(dataset))
                    batch_indices = list(range(start_idx, end_idx))
                    all_indices.extend(batch_indices)
                    
                    # Forward pass
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Calcular probabilidade da classe correta
                    correct_probs = probs[range(len(targets)), targets]
                    
                    # Converter para CPU e NumPy
                    difficulty_scores.extend(correct_probs.cpu().numpy())
            
            # Normalizar scores (0 = mais difícil, 1 = mais fácil)
            difficulty_scores = np.array(difficulty_scores)
            
            # Registrar evolução da dificuldade
            difficulty_evolution.append(difficulty_scores.mean())
            
            # === FASE 2: SELEÇÃO DE SUBSET ===
            # Calcular tamanho do subset nesta época (crescente)
            ratio = initial_ratio + (final_ratio - initial_ratio) * (epoch / (epochs - 1))
            subset_size = int(len(dataset) * ratio)
            subset_sizes.append(subset_size)
            
            # Número de amostras difíceis e aleatórias
            num_hard = int(subset_size * hard_fraction)
            num_random = subset_size - num_hard
            
            # Selecionar exemplos mais difíceis
            hard_indices = np.argsort(difficulty_scores)[:num_hard]
            
            # Selecionar exemplos aleatórios dos restantes
            remaining_indices = np.setdiff1d(np.arange(len(dataset)), hard_indices)
            random_indices = np.random.choice(remaining_indices, num_random, replace=False)
            
            # Combinar índices
            selected_indices = np.concatenate([hard_indices, random_indices])
            
            logger.info(f"Selecionadas {len(selected_indices)} amostras ({ratio*100:.1f}%)")
            logger.info(f"  - {num_hard} amostras difíceis, {num_random} amostras aleatórias")
        else:
            # Na primeira época, usar subset aleatório inicial
            initial_size = int(len(dataset) * initial_ratio)
            selected_indices = np.random.choice(
                np.arange(len(dataset)), initial_size, replace=False
            )
            subset_sizes.append(initial_size)
            logger.info(f"Época inicial: selecionadas {len(selected_indices)} amostras aleatórias")
        
        # Criar subset e dataloader para treinamento
        subset = Subset(dataset, selected_indices)
        train_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        
        # === FASE 3: TREINAMENTO COM SUBSET ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward e otimização
            loss.backward()
            optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log de progresso
            if batch_idx % 20 == 0:
                logger.debug(f"Treinamento: Batch {batch_idx}/{len(train_loader)}")
        
        # Calcular métricas de treino
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        logger.info(f"Treino => Perda: {train_loss:.4f}, Acurácia: {train_acc:.2f}%")
        
        # === FASE 4: VALIDAÇÃO ===
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Estatísticas
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calcular métricas de validação
        val_loss = running_loss / len(test_loader)
        val_acc = 100.0 * correct / total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Validação => Perda: {val_loss:.4f}, Acurácia: {val_acc:.2f}%")
        
        # Atualizar TensorBoard
        writer.add_scalar("DifficultyTraining/train_loss", train_loss, epoch)
        writer.add_scalar("DifficultyTraining/val_loss", val_loss, epoch)
        writer.add_scalar("DifficultyTraining/train_accuracy", train_acc, epoch)
        writer.add_scalar("DifficultyTraining/val_accuracy", val_acc, epoch)
        writer.add_scalar("DifficultyTraining/subset_size", len(selected_indices), epoch)
        
        if len(difficulty_evolution) > 0:
            writer.add_scalar("DifficultyTraining/mean_difficulty", difficulty_evolution[-1], epoch)
        
        # Ajustar learning rate se houver scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
            # Registrar learning rate atual
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("DifficultyTraining/learning_rate", current_lr, epoch)
        
        # Salvar melhor modelo
        if val_acc > best_acc:
            logger.info(f"Acurácia de validação melhorou: {best_acc:.2f}% → {val_acc:.2f}%")
            best_acc = val_acc
            
            # Salvar checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, f"models/difficulty_sampling_best.pth")
    
    # Tempo total de treinamento
    total_time = time.time() - start_time
    logger.info(f"Treinamento concluído em {total_time/60:.2f} minutos")
    logger.info(f"Melhor acurácia de validação: {best_acc:.2f}%")
    
    # Criar gráficos finais
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, epochs+1)
    
    # Gráfico de perda
    ax1.plot(epochs_range, train_losses, 'b-', label='Treino')
    ax1.plot(epochs_range, val_losses, 'r-', label='Validação')
    ax1.set_title('Curvas de Perda')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de acurácia
    ax2.plot(epochs_range, train_accs, 'b-', label='Treino')
    ax2.plot(epochs_range, val_accs, 'r-', label='Validação')
    ax2.set_title('Curvas de Acurácia')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico de tamanho do subset
    ax3.plot(epochs_range, subset_sizes, 'g-')
    ax3.set_title('Tamanho do Subset')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Número de Amostras')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    writer.add_figure("DifficultyTraining/training_curves", fig, 0)
    plt.savefig("results/difficulty_sampling_curves.png")
    plt.close(fig)
    
    # Fechar TensorBoard writer
    writer.close()
    
    # Carregar melhor modelo
    checkpoint = torch.load("models/difficulty_sampling_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_accuracy': best_acc,
        'difficulty_evolution': difficulty_evolution,
        'subset_sizes': subset_sizes
    }

def compare_sampling_methods(dataset, test_dataset, num_samples, model_builder, 
                           epochs=5, batch_size=32, device=None):
    """
    Compara diferentes métodos de amostragem para a mesma tarefa.
    
    Args:
        dataset: Dataset completo de treinamento
        test_dataset: Dataset de teste para avaliação
        num_samples: Número de amostras para cada método
        model_builder: Função que retorna uma instância de modelo
        epochs: Número de épocas para treinamento
        batch_size: Tamanho do batch
        device: Dispositivo para processamento
        
    Returns:
        dict: Resultados comparativos
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Comparando métodos de amostragem com {num_samples} amostras...")
    
    # Métodos de amostragem a serem comparados
    sampling_methods = {
        "random": lambda: np.random.choice(len(dataset), num_samples, replace=False),
        "coreset": lambda: select_diverse_coreset(dataset, num_samples),
        "kmeans": lambda: select_coreset_kmeans(dataset, num_samples)
    }
    
    # Dataloader para teste
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Resultados para cada método
    results = {}
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir="tensorboard_logs/sampling_comparison")
    
    # Para cada método de amostragem
    for method_name, sampler in sampling_methods.items():
        logger.info(f"Testando método: {method_name}")
        
        try:
            # Selecionar amostras
            logger.info(f"Selecionando amostras com método {method_name}...")
            indices = sampler()
            
            # Criar subset e dataloader
            subset = Subset(dataset, indices)
            train_loader = DataLoader(
                subset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            
            # Criar novo modelo e otimizador
            model = model_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()
            
            # Métricas
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            # Training loop
            for epoch in range(epochs):
                # === TREINO ===
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Zerar gradientes
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward e otimização
                    loss.backward()
                    optimizer.step()
                    
                    # Estatísticas
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                # Métricas de treino
                train_loss = running_loss / len(train_loader)
                train_acc = 100.0 * correct / total
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # === VALIDAÇÃO ===
                model.eval()
                running_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Estatísticas
                        running_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                # Métricas de validação
                val_loss = running_loss / len(test_loader)
                val_acc = 100.0 * correct / total
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Atualizar TensorBoard
                writer.add_scalar(f"Comparison/{method_name}_train_loss", train_loss, epoch)
                writer.add_scalar(f"Comparison/{method_name}_val_loss", val_loss, epoch)
                writer.add_scalar(f"Comparison/{method_name}_train_acc", train_acc, epoch)
                writer.add_scalar(f"Comparison/{method_name}_val_acc", val_acc, epoch)
                
                # Atualizar scheduler
                scheduler.step()
                
                logger.info(f"{method_name} - Época {epoch+1}/{epochs}, "
                          f"Treino: {train_acc:.2f}%, Val: {val_acc:.2f}%")
            
            # Armazenar resultados
            results[method_name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'final_acc': val_accs[-1]
            }
            
        except Exception as e:
            logger.error(f"Erro durante avaliação do método {method_name}: {str(e)}")
    
    # Criar visualização comparativa
    if results:
        # Comparação de acurácia de validação
        plt.figure(figsize=(12, 6))
        for method, metrics in results.items():
            plt.plot(range(1, epochs+1), metrics['val_accs'], label=f"{method}")
        
        plt.title('Comparação de Métodos de Amostragem')
        plt.xlabel('Época')
        plt.ylabel('Acurácia de Validação (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("results/sampling_comparison.png")
        writer.add_figure("Comparison/methods", plt.gcf(), 0)
        plt.close()
        
        # Comparação final
        methods = list(results.keys())
        final_accs = [results[m]['val_accs'][-1] for m in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, final_accs, color='skyblue')
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, final_accs):
            plt.text(bar.get_x() + bar.get_width()/2., acc + 1,
                    f'{acc:.2f}%', ha='center', va='bottom')
        
        plt.title(f'Acurácia Final por Método de Amostragem ({num_samples} amostras)')
        plt.xlabel('Método')
        plt.ylabel('Acurácia de Validação (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/sampling_comparison_final.png")
        writer.add_figure("Comparison/final", plt.gcf(), 0)
        plt.close()
    
    # Fechar writer
    writer.close()
    
    return results