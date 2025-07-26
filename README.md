# 🧾 FiscAssistant – Chatbot Fiscal Multilingue 🇫🇷🇬🇧🇹🇳🇸🇦

**FiscAssistant** est un assistant fiscal intelligent, multilingue, et 100% local. Il guide les utilisateurs dans leurs obligations fiscales, les aide à remplir des déclarations, effectue des calculs comme la TVA ou l’IRPP, et offre des rappels personnalisés. Ce projet est développé avec **Flask** et intègre le modèle **Nous-Hermes-2** pour un raisonnement avancé en langage naturel.

---

## 🚀 Fonctionnalités principales

### A. Accueil & Orientation

* Répond aux salutations dans plusieurs langues (Bonjour, Hello, مرحبا)
* Présente ses fonctions automatiquement
* Pose des questions pour comprendre le besoin fiscal de l’utilisateur
* Identifie le profil fiscal : particulier, société, auto-entrepreneur…

### B. Déclarations Fiscales

* Donne les **dates limites** mensuelles, trimestrielles et annuelles
* Explique les types de déclarations : **TVA, IRPP, IS, CNSS**
* Aide à remplir les formulaires
* Vérifie les retards de déclaration

### C. TVA & Régimes fiscaux

* Donne les taux de TVA selon le secteur d’activité
* Calcule automatiquement la TVA à payer ou à récupérer
* Explique les régimes (réel, forfaitaire, exonéré)
* Conseille sur le régime le plus adapté

### D. Infractions & Risques

* Explique les pénalités de retard ou de non-déclaration
* Alerte sur les sanctions
* Simule un contrôle fiscal
* Donne les recours possibles

### E. Simulation de Rentabilité

* Calcule le bénéfice net après impôts et charges sociales
* Calcule le **seuil de rentabilité**
* Évalue la viabilité d’un projet
* Simule l’IRPP et la CNSS à partir d’un revenu

### F. Création d’Entreprise

* Guide dans le **choix du statut juridique** (SARL, SUARL…)
* Donne les étapes de création d’une entreprise en Tunisie
* Explique les obligations fiscales associées

### G. Rappels & Notifications

* Envoie des rappels de déclaration ou d’échéance
* Alerte sur les changements dans la législation fiscale

### H. Formation & Éducation Fiscale

* Explique les **concepts fiscaux** de façon simple
* Fournit des définitions, liens utiles, quiz interactifs

### I. Multilingue & Localisé

* Répond en **français, anglais, arabe standard**, et **dialecte tunisien**
* Peut changer de langue à la demande de l’utilisateur

---

## 🧠 Technologies utilisées

| Composant                     | Description                                                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 🧠 **LLM**                    | [Nous-Hermes-2](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral) – modèle local, multilingue, puissant |
| 🧪 **Serveur**                | [Flask](https://flask.palletsprojects.com/) – API backend simple et rapide                                       |
| 📦 **Frontend**               | Interface utilisateur (UI) moderne en HTML/CSS/JS ou avec un framework comme React                               |
| 🗣️ **Langchain** (optionnel) | Pour la gestion de la mémoire, des outils, et des chaînes de traitement                                          |
| 📐 **Custom Logic**           | Fonctions de calcul fiscal personnalisées (TVA, IRPP, CNSS, etc.)                                                |

---

## 🖥️ Déploiement local

1. **Cloner le projet**

```bash
git clone https://github.com/votre-utilisateur/fiscassistant.git
cd fiscassistant
```

2. **Installer les dépendances Python**

```bash
pip install -r requirements.txt
```

3. **Télécharger le modèle Nous-Hermes-2 via Ollama**

```bash
ollama run nous-hermes-2-mixtral
```

4. **Lancer le serveur Flask**

```bash
python app.py
```

5. **Accéder à l’interface**
   Visitez `http://localhost:5000` dans votre navigateur.

---

## 📜 Licence

Ce projet est open-source pour usage éducatif ou personnel. Pour un usage commercial, veuillez consulter les conditions de licence du modèle LLM utilisé.