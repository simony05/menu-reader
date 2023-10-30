import React, { useState } from 'react';
import axios from 'axios';

function App() {
 const [selectedFile, setSelectedFile] = useState(null);
 const [isSelected, setIsSelected] = useState(false);

 const changeHandler = (event) => {
    setSelectedFile(event.target.files[0]);
    setIsSelected(true);
 };

 const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      await axios.post('http://localhost:5000/', formData)
      .then(response =>
        console.log(response))
    } catch (error) {
      console.log(error);
    }
 };

 return (
    <div className="App">
      <form onSubmit={handleSubmit}>
        <input type="file" name="file" accept="image/*" onChange={changeHandler} />
        {isSelected ? <div>{selectedFile.name}</div> : null}
        <button type="submit">Upload</button>
      </form>
    </div>
 );
}

export default App;