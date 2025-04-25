#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para demonstrar a detecção de imagens duplicadas usando imagehash.

Este script analisa um diretório de imagens, calcula hashes perceptuais
para cada imagem e identifica grupos de imagens similares baseados em um
limiar de similaridade configurável.

Uso:
    python duplicate_detector_demo.py --dir /caminho/para/dataset --threshold 3
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from tqdm import tqdm
import logging
import sys
from pathlib import Path
import glob

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("duplicate_detection.log")
    ]
)

logger = logging.getLogger("duplicate_detector")

def scan_image_directory(directory):
    """
    Escaneia um diretório e cria um DataFrame com informações sobre as imagens.
    
    Args:
        directory: Caminho para o diretório contendo imagens
        
    Returns:
        DataFrame com metadados das imagens
    """
    logger.info(f"Escaneando diretório: {directory}")
    
    data = []
    corrupted_images = []
    
    # Encontrar todos os arquivos de imagem (extensões comuns)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    logger.info(f"Encontradas {len(image_files)} imagens")
    
    # Processar cada imagem
    for img_path in tqdm(image_files, desc="Processando imagens"):
        img_corrupted = False
        img_hash = None
        width, height, channels = None, None, None
        img_format = None
        
        try:
            # Tenta abrir a imagem para verificar se está corrompida
            with Image.open(img_path) as img:
                # Obter dimensões
                width, height = img.size
                
                # Obter formato da imagem
                img_format = os.path.splitext(img_path)[1][1:].lower()
                
                # Calcular o hash perceptual
                img_hash = str(imagehash.phash(img))
                
                # Verificar se tem canais de cor
                if img.mode == 'RGB':
                    channels = 3
                elif img.mode == 'RGBA':
                    channels = 4
                elif img.mode == 'L':
                    channels = 1
                else:
                    channels = 0  # Desconhecido
                
        except Exception as e:
            img_corrupted = True
            corrupted_images.append((img_path, str(e)))
            logger.warning(f"Erro ao processar {img_path}: {str(e)}")
        
        # Adiciona os metadados ao dataframe
        data.append({
            'image_path': img_path,
            'image_format': img_format,
            'width': width,
            'height': height,
            'channels': channels,
            'image_hash': img_hash,
            'corrupted': img_corrupted
        })
    
    df = pd.DataFrame(data)
    
    # Logar estatísticas
    logger.info(f"Total de imagens processadas: {len(df)}")
    logger.info(f"Total de imagens corrompidas: {len(corrupted_images)}")
    
    return df, corrupted_images

def detect_duplicate_images(dataframe, hash_diff_threshold=3):
    """
    Detecta imagens duplicadas ou quase idênticas no dataset usando imagehash.
    
    Args:
        dataframe: DataFrame pandas com coluna 'image_path' e 'image_hash'
        hash_diff_threshold: Limiar para considerar imagens similares (menor = mais restritivo)
        
    Returns:
        DataFrame com grupos de imagens duplicadas/similares
    """
    logger.info(f"Iniciando detecção de imagens duplicadas com limiar {hash_diff_threshold}...")
    
    # Verificar se a coluna 'image_hash' existe
    if 'image_hash' not in dataframe.columns:
        logger.error("Coluna 'image_hash' não encontrada no DataFrame")
        return None
    
    # Remover imagens corrompidas ou sem hash
    valid_df = dataframe[~dataframe['corrupted'] & ~dataframe['image_hash'].isna()]
    logger.info(f"Analisando {len(valid_df)} imagens válidas")
    
    # Criar dicionário para agrupar imagens por hash
    similar_images = {}
    duplicates_count = 0
    
    # Para cada imagem no dataframe
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Procurando duplicatas"):
        img_path = row['image_path']
        img_hash = row['image_hash']
        
        # Verificar similaridade com hashes já processados
        found_similar = False
        for existing_hash in list(similar_images.keys()):
            # Converter strings para objetos hash, se necessário
            img_hash_obj = imagehash.hex_to_hash(img_hash) if isinstance(img_hash, str) else img_hash
            existing_hash_obj = imagehash.hex_to_hash(existing_hash) if isinstance(existing_hash, str) else existing_hash
            
            # Calcular diferença de hash
            hash_diff = abs(img_hash_obj - existing_hash_obj)
            
            if hash_diff <= hash_diff_threshold:
                similar_images[existing_hash].append((img_path, hash_diff))
                found_similar = True
                duplicates_count += 1
                break
        
        if not found_similar:
            similar_images[img_hash] = [(img_path, 0)]  # Primeira imagem deste hash
    
    # Filtrar apenas grupos com mais de uma imagem (duplicatas)
    duplicate_groups = {h: imgs for h, imgs in similar_images.items() if len(imgs) > 1}
    
    # Criar dataframe com resultados
    result_data = []
    for hash_val, image_list in duplicate_groups.items():
        for img_path, diff in image_list:
            result_data.append({
                'hash': str(hash_val),
                'image_path': img_path,
                'similarity_diff': diff,
                'group_size': len(image_list)
            })
    
    # Criar dataframe com o resultado
    if result_data:
        result_df = pd.DataFrame(result_data)
        logger.info(f"Encontradas {duplicates_count} imagens similares em {len(duplicate_groups)} grupos")
        return result_df
    else:
        logger.info("Nenhuma imagem duplicada encontrada")
        return pd.DataFrame(columns=['hash', 'image_path', 'similarity_diff', 'group_size'])

def visualize_duplicates(duplicates_df, num_groups=3, output_dir="results"):
    """
    Visualiza grupos de imagens duplicadas.
    
    Args:
        duplicates_df: DataFrame com informações sobre duplicatas
        num_groups: Número de grupos a visualizar
        output_dir: Diretório para salvar as visualizações
    """
    if duplicates_df is None or len(duplicates_df) == 0:
        logger.info("Sem duplicatas para visualizar")
        return
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Obter grupos de hashes únicos
    groups = duplicates_df['hash'].unique()
    
    # Limitar ao número de grupos solicitado
    groups = groups[:min(num_groups, len(groups))]
    
    # Para cada grupo
    for group_idx, group_hash in enumerate(groups):
        # Filtrar imagens deste grupo
        group_images = duplicates_df[duplicates_df['hash'] == group_hash]
        
        # Ordenar por diferença (as mais similares primeiro)
        group_images = group_images.sort_values('similarity_diff')
        
        # Número de imagens neste grupo (limitado a 6 para visualização)
        num_images = min(6, len(group_images))
        
        # Calcular layout da figura
        rows = 1 if num_images <= 3 else 2
        cols = (num_images + rows - 1) // rows
        
        plt.figure(figsize=(cols * 5, rows * 5))
        
        # Mostrar cada imagem
        for i, (_, row) in enumerate(group_images.head(num_images).iterrows()):
            plt.subplot(rows, cols, i + 1)
            
            try:
                img = Image.open(row['image_path'])
                plt.imshow(img)
                plt.title(f"Diff: {row['similarity_diff']}")
                plt.axis('off')
                
                # Mostrar caminho resumido da imagem
                img_path = Path(row['image_path']).name
                plt.xlabel(img_path, fontsize=8)
            except Exception as e:
                plt.text(0.5, 0.5, f"Erro: {str(e)}", ha='center', va='center', transform=plt.gca().transAxes)
        
        # Título global
        plt.suptitle(f"Grupo {group_idx+1}: Hash {group_hash[:8]}... ({len(group_images)} imagens)")
        plt.tight_layout()
        
        # Salvar figura
        output_path = os.path.join(output_dir, f"duplicate_group_{group_idx+1}.png")
        plt.savefig(output_path)
        logger.info(f"Visualização salva em {output_path}")
        plt.close()
    
    # Criar resumo
    plt.figure(figsize=(10, 6))
    
    # Contar tamanhos dos grupos
    group_sizes = duplicates_df.groupby('hash').size().value_counts().sort_index()
    
    # Plotar histograma dos tamanhos dos grupos
    plt.bar(group_sizes.index, group_sizes.values)
    plt.xlabel("Tamanho do grupo")
    plt.ylabel("Número de grupos")
    plt.title("Distribuição de tamanhos de grupos de imagens similares")
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar rótulos nas barras
    for x, y in zip(group_sizes.index, group_sizes.values):
        plt.text(x, y + 0.1, str(y), ha='center')
    
    # Salvar resumo
    summary_path = os.path.join(output_dir, "duplicate_summary.png")
    plt.savefig(summary_path)
    logger.info(f"Resumo salvo em {summary_path}")
    plt.close()

def parse_arguments():
    """Processa argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="Detector de imagens duplicadas usando imagehash")
    
    parser.add_argument('--dir', type=str, required=True,
                        help='Diretório contendo as imagens a analisar')
    parser.add_argument('--threshold', type=int, default=3,
                        help='Limiar de diferença para considerar imagens como similares (menor = mais restritivo)')
    parser.add_argument('--num_groups', type=int, default=3,
                        help='Número de grupos de duplicatas a visualizar')
    parser.add_argument('--output', type=str, default="results",
                        help='Diretório para salvar os resultados')
    
    return parser.parse_args()

def main():
    """Função principal do script"""
    # Processar argumentos
    args = parse_arguments()
    
    # Escanear diretório de imagens
    df, corrupted = scan_image_directory(args.dir)
    
    # Detectar duplicatas
    duplicates_df = detect_duplicate_images(df, hash_diff_threshold=args.threshold)
    
    # Visualizar duplicatas
    if duplicates_df is not None and len(duplicates_df) > 0:
        visualize_duplicates(duplicates_df, num_groups=args.num_groups, output_dir=args.output)
        
        # Salvar relatório de duplicatas
        output_csv = os.path.join(args.output, "duplicate_images.csv")
        duplicates_df.to_csv(output_csv, index=False)
        logger.info(f"Relatório salvo em {output_csv}")
        
        # Exibir estatísticas
        print("\n===== RESUMO DE DUPLICATAS =====")
        print(f"Total de imagens analisadas: {len(df)}")
        print(f"Imagens corrompidas: {len(corrupted)}")
        print(f"Grupos de duplicatas encontrados: {duplicates_df['hash'].nunique()}")
        print(f"Total de imagens duplicadas: {len(duplicates_df)}")
        
        # Sugestão para limpeza
        print("\nSugestão: Para limpar o dataset, você pode manter apenas a primeira imagem de cada grupo.")
        print("          Isso reduziria o tamanho em aproximadamente", 
             f"{(len(duplicates_df) - duplicates_df['hash'].nunique()) / len(df) * 100:.1f}%")
    else:
        print("Nenhuma duplicata encontrada com o limiar atual.")
        print(f"Dica: Tente aumentar o limiar (--threshold {args.threshold + 1}) para encontrar imagens mais distintas.")

if __name__ == "__main__":
    main()