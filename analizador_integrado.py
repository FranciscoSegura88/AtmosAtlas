#!/usr/bin/env python3
"""
analizador_integrado.py

Versión integrada del analizador climático con ML avanzado
Combina el analizador original con los nuevos modelos avanzados
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Importar analizador original
from analizador import ClimateAnalyzer

# Importar predictor avanzado
from advanced_ml_predictor import (
    AdvancedClimatePredictor, 
    ClimateDataPreprocessor,
    integrate_with_analyzer
)


class IntegratedClimateAnalyzer:
    """
    Analizador climático integrado que combina:
    - Análisis estadístico tradicional
    - Machine Learning básico (Logistic Regression, Random Forest)
    - Machine Learning avanzado (XGBoost, LightGBM, ensemble)
    """
    
    def __init__(self):
        self.basic_analyzer = ClimateAnalyzer()
        self.advanced_predictor = AdvancedClimatePredictor()
        self.results = {}
        
    def analyze_comprehensive(self, 
                             lat: float, 
                             lon: float,
                             target_date: str,
                             start_date: str = "1990-01-01",
                             end_date: str = "2024-12-31",
                             window_days: int = 7,
                             rain_threshold: float = 0.5,
                             use_advanced_ml: bool = True,
                             save_models: bool = False) -> dict:
        """
        Realiza análisis comprehensivo combinando todos los métodos
        
        Args:
            lat: Latitud
            lon: Longitud
            target_date: Fecha objetivo (YYYY-MM-DD)
            start_date: Inicio datos históricos
            end_date: Fin datos históricos
            window_days: Ventana temporal (±días)
            rain_threshold: Umbral de lluvia (mm)
            use_advanced_ml: Usar modelos ML avanzados
            save_models: Guardar modelos entrenados
            
        Returns:
            Dict con todos los análisis
        """
        print(f"\n{'='*70}")
        print(f"🌍 ANÁLISIS CLIMÁTICO INTEGRADO - NASA Space Apps Challenge")
        print(f"{'='*70}")
        print(f"\n📍 Ubicación: ({lat}, {lon})")
        print(f"📅 Fecha objetivo: {target_date}")
        print(f"📊 Datos históricos: {start_date} a {end_date}")
        print(f"⏱️  Ventana temporal: ±{window_days} días\n")
        
        # 1. Descargar datos
        print("📥 Descargando datos de NASA POWER...")
        self.basic_analyzer.download_data(lat, lon, start_date, end_date)
        
        # 2. Extraer ventana histórica
        print(f"🔍 Extrayendo ventana histórica para {target_date}...")
        window_df = self.basic_analyzer.get_historical_window(target_date, window_days)
        
        # 3. Análisis estadístico básico
        print("📈 Calculando estadísticas históricas...")
        stats = self.basic_analyzer.calculate_probabilities(
            window_df,
            rain_threshold=rain_threshold
        )
        
        self.results['statistics'] = stats
        self.results['target_date'] = target_date
        self.results['location'] = {'lat': lat, 'lon': lon}
        
        # 4. ML básico (del analizador original)
        print("🤖 Entrenando modelos ML básicos...")
        ml_basic = self.basic_analyzer.train_ml_models(window_df, rain_threshold)
        
        if ml_basic:
            basic_pred = self.basic_analyzer.predict_for_date(
                target_date, 
                window_df, 
                ml_basic
            )
            self.results['ml_basic'] = {
                'prediction': basic_pred,
                'metrics': ml_basic['metrics']
            }
        
        # 5. ML avanzado (nuevo)
        if use_advanced_ml:
            print("🚀 Entrenando modelos ML avanzados (ensemble)...")
            try:
                advanced_results = integrate_with_analyzer(
                    self.basic_analyzer,
                    self.advanced_predictor,
                    window_df,
                    target_date,
                    rain_threshold
                )
                self.results['ml_advanced'] = advanced_results
                
                if save_models:
                    model_path = self.advanced_predictor.save_models(
                        prefix=f"model_{lat}_{lon}_{target_date}"
                    )
                    print(f"💾 Modelos guardados en: {model_path}")
                    
            except Exception as e:
                print(f"⚠️  Error en ML avanzado: {e}")
                self.results['ml_advanced'] = {'error': str(e)}
        
        # 6. Generar recomendación final
        self._generate_final_recommendation()
        
        return self.results
    
    def _generate_final_recommendation(self):
        """Genera recomendación final combinando todos los análisis"""
        recommendations = []
        confidence_scores = []
        
        # Probabilidad estadística
        stat_prob = self.results['statistics'].get('rain_probability', 0)
        recommendations.append({
            'method': 'Estadística Histórica',
            'probability': stat_prob,
            'will_rain': stat_prob > 0.5
        })
        confidence_scores.append(0.7)  # Confianza media
        
        # ML básico
        if 'ml_basic' in self.results:
            ml_prob = self.results['ml_basic']['prediction'].get('ensemble_prob', 0)
            recommendations.append({
                'method': 'ML Básico (Ensemble)',
                'probability': ml_prob,
                'will_rain': ml_prob > 0.5
            })
            confidence_scores.append(0.8)
        
        # ML avanzado
        if 'ml_advanced' in self.results and 'error' not in self.results['ml_advanced']:
            adv_prob = self.results['ml_advanced']['advanced_ml']['precipitation_mm']
            adv_will_rain = self.results['ml_advanced']['advanced_ml']['will_rain']
            recommendations.append({
                'method': 'ML Avanzado (XGBoost+LightGBM)',
                'probability': adv_prob / 10,  # Normalizar a 0-1
                'will_rain': adv_will_rain
            })
            confidence_scores.append(0.9)
        
        # Calcular recomendación final (weighted voting)
        total_weight = sum(confidence_scores)
        weighted_prob = sum(
            rec['probability'] * weight 
            for rec, weight in zip(recommendations, confidence_scores)
        ) / total_weight
        
        # Consenso
        votes_rain = sum(1 for rec in recommendations if rec['will_rain'])
        consensus = votes_rain / len(recommendations)
        
        self.results['final_recommendation'] = {
            'will_rain': weighted_prob > 0.5,
            'probability': weighted_prob,
            'consensus': consensus,
            'confidence': 'high' if consensus >= 0.8 or consensus <= 0.2 else 'medium',
            'methods_used': [rec['method'] for rec in recommendations],
            'individual_predictions': recommendations
        }
    
    def print_results(self):
        """Imprime resultados en formato legible"""
        print(f"\n{'='*70}")
        print(f"📊 RESULTADOS DEL ANÁLISIS")
        print(f"{'='*70}\n")
        
        # Estadísticas
        stats = self.results['statistics']
        print(f"📈 Estadísticas Históricas:")
        print(f"   • Muestras: {stats['n_samples']} días de {stats['n_years']} años")
        print(f"   • Probabilidad de lluvia: {stats['rain_probability']*100:.1f}%")
        print(f"   • Temperatura promedio: {stats['avg_temp']:.1f}°C (±{stats['temp_std']:.1f}°C)")
        
        if 'avg_humidity' in stats:
            print(f"   • Humedad promedio: {stats['avg_humidity']:.1f}%")
        
        # ML Básico
        if 'ml_basic' in self.results:
            print(f"\n🤖 Machine Learning Básico:")
            pred = self.results['ml_basic']['prediction']
            metrics = self.results['ml_basic']['metrics']
            print(f"   • Probabilidad: {pred.get('ensemble_prob', 0)*100:.1f}%")
            print(f"   • Precisión Logistic Regression: {metrics['logreg_acc']*100:.1f}%")
            print(f"   • Precisión Random Forest: {metrics['rf_acc']*100:.1f}%")
        
        # ML Avanzado
        if 'ml_advanced' in self.results and 'error' not in self.results['ml_advanced']:
            print(f"\n🚀 Machine Learning Avanzado:")
            adv = self.results['ml_advanced']['advanced_ml']
            print(f"   • Precipitación predicha: {adv['precipitation_mm']:.2f} mm")
            print(f"   • ¿Lloverá? {'SÍ' if adv['will_rain'] else 'NO'}")
            print(f"   • Intervalo confianza: [{adv['confidence_interval']['lower']:.2f}, "
                  f"{adv['confidence_interval']['upper']:.2f}] mm")
            print(f"   • Nivel de confianza: {adv['confidence_level'].upper()}")
            
            # Mostrar predicciones individuales
            print(f"\n   Modelos individuales:")
            for model, value in adv['individual_models'].items():
                print(f"      - {model}: {value:.2f} mm")
        
        # Recomendación final
        print(f"\n{'='*70}")
        print(f"🎯 RECOMENDACIÓN FINAL")
        print(f"{'='*70}")
        
        final = self.results['final_recommendation']
        
        # Emoji según predicción
        rain_emoji = "🌧️" if final['will_rain'] else "☀️"
        
        print(f"\n{rain_emoji}  {'LLOVERÁ' if final['will_rain'] else 'NO LLOVERÁ'}")
        print(f"\n   • Probabilidad combinada: {final['probability']*100:.1f}%")
        print(f"   • Consenso entre métodos: {final['consensus']*100:.0f}%")
        print(f"   • Nivel de confianza: {final['confidence'].upper()}")
        print(f"   • Métodos utilizados: {len(final['methods_used'])}")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analizador climático integrado con ML avanzado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Análisis completo con ML avanzado
  python analizador_integrado.py \\
    --lat 20.67 --lon -103.35 \\
    --target_date 2025-08-15 \\
    --start 1990-01-01 --end 2024-12-31

  # Solo estadísticas y ML básico (más rápido)
  python analizador_integrado.py \\
    --lat 19.43 --lon -99.13 \\
    --target_date 2025-12-25 \\
    --no-advanced-ml

  # Con guardado de modelos
  python analizador_integrado.py \\
    --lat 20.67 --lon -103.35 \\
    --target_date 2025-08-15 \\
    --save-models
        """
    )
    
    # Ubicación y fecha
    parser.add_argument('--lat', type=float, required=True, 
                       help='Latitud (-90 a 90)')
    parser.add_argument('--lon', type=float, required=True, 
                       help='Longitud (-180 a 180)')
    parser.add_argument('--target_date', required=True, 
                       help='Fecha objetivo (YYYY-MM-DD)')
    
    # Datos históricos
    parser.add_argument('--start', default='1990-01-01',
                       help='Fecha inicio datos (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31',
                       help='Fecha fin datos (YYYY-MM-DD)')
    
    # Parámetros
    parser.add_argument('--window_days', type=int, default=7,
                       help='Ventana temporal (±días)')
    parser.add_argument('--rain_threshold', type=float, default=0.5,
                       help='Umbral de lluvia (mm)')
    
    # Opciones
    parser.add_argument('--no-advanced-ml', action='store_true',
                       help='No usar ML avanzado (solo básico)')
    parser.add_argument('--save-models', action='store_true',
                       help='Guardar modelos entrenados')
    parser.add_argument('--output', help='Archivo JSON de salida')
    
    args = parser.parse_args()
    
    try:
        # Crear analizador integrado
        analyzer = IntegratedClimateAnalyzer()
        
        # Realizar análisis
        results = analyzer.analyze_comprehensive(
            lat=args.lat,
            lon=args.lon,
            target_date=args.target_date,
            start_date=args.start,
            end_date=args.end,
            window_days=args.window_days,
            rain_threshold=args.rain_threshold,
            use_advanced_ml=not args.no_advanced_ml,
            save_models=args.save_models
        )
        
        # Mostrar resultados
        analyzer.print_results()
        
        # Guardar a JSON si se especificó
        if args.output:
            import json
            
            # Convertir numpy types a Python types
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                return obj
            
            results_clean = convert_types(results)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results_clean, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()