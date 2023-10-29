import React, { Component } from 'react';

class App extends Component {
  constructor() {
    super();
    this.state = {
      postData: {
        key1: 'value1',
        key2: 'value2',
      },
      responseMessage: '',
    };
  }

  handleSubmit = () => {
    const { postData } = this.state;

    fetch('http://localhost:5000/api', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(postData),
    })
      .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error('Network response was not ok');
        }
      })
      .then(data => {
        this.setState({ responseMessage: data.message });
      })
      .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
      });
  };

  render() {
    const { responseMessage } = this.state;

    return (
      <div>
        <h1>React POST Request to Flask</h1>
        <button onClick={this.handleSubmit}>Submit POST Request</button>
        <p>Response: {responseMessage}</p>
      </div>
    );
  }
}

export default App;
