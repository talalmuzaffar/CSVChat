# CSVChat with AI 🤖

A powerful Streamlit-based web application that enables users to interact with CSV data using natural language queries. Powered by OpenAI's GPT models and LangChain, this tool makes data analysis accessible and intuitive through conversation.

## 🌟 Features

- **Intuitive Data Interaction**: Chat with your data using natural language
- **Secure API Handling**: Safe and encrypted handling of API keys
- **CSV Support**: Upload and analyze any CSV file
- **Smart Schema Detection**: Automatic detection and display of data structure
- **Interactive Analysis**: Real-time data exploration and visualization
- **Chat History**: Preserve your conversation and analysis flow
- **User-Friendly Interface**: Clean and modern Streamlit-based UI
- **Pandas Integration**: Powered by LangChain's Pandas Agent for robust data operations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/talalmuzaffar/CSVChat.git
   cd CSVChat
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 💡 Usage

1. Launch the application using the command above
2. Upload your CSV file through the web interface
3. View the automatically detected schema and data preview
4. Start chatting with your data using natural language queries
5. Explore insights and visualizations generated from your questions

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: 
  - OpenAI GPT models
  - LangChain framework
  - Pandas for data manipulation
- **Data Visualization**: Streamlit native charts

## 📝 Example Queries

- "Show me the top 5 rows of the dataset"
- "What is the average value in column X?"
- "Create a bar chart of Y grouped by Z"
- "Find correlations between columns A and B"
- "Give me a summary of the data"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Talal Muzaffar - *Initial work* - [GitHub](https://github.com/talalmuzaffar)

## 🙏 Acknowledgments

- LangChain for the excellent framework
- Streamlit for making web apps simple and beautiful

## 📞 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/talalmuzaffar/CSVChat/issues) on GitHub.
