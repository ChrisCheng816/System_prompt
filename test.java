public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        // Clear the existing buffer
        this.buffer = new byte[0];
        // Use a ByteArrayOutputStream to dynamically grow the buffer as we read
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
            baos.write(buffer, 0, bytesRead);
        }
        // Set the buffer to the contents of the ByteArrayOutputStream
        this.buffer = baos.toByteArray();
        // Reset the read pointer to the start of the byte array
        this.readPointer = 0;
    } catch (IOException e) {
        throw new IOException("Error reading from InputStream", e);
    }
}

public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        byte[] buffer = new byte[in.available()];
        int bytesRead = in.read(buffer);
        if (bytesRead != buffer.length) {
            throw new IOException("Failed to read the expected number of bytes");
        }
        // Assuming there's a byte array field named 'buffer' in the class
        this.buffer = buffer;
    } finally {
        in.close();
    }
}

public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        byte[] buffer = new byte[in.available()];
        int bytesRead = in.read(buffer);
        if (bytesRead != buffer.length) {
            throw new IOException("Failed to read the expected number of bytes");
        }
        // Assuming there's a byte array to fill in the ClassFileBuffer
        this.buffer = buffer;
        this.position = 0; // Reset the read pointer to the start
    } finally {
        in.close();
    }
}

public void readFrom(final InputStream in) throws IOException {
    buffer.clear();
    int read;
    while ((read = in.read(buffer.array(), buffer.position(), buffer.remaining())) != -1) {
        buffer.position(buffer.position() + read);
    }
    buffer.flip();
}

public void readFrom(final InputStream in) throws IOException {
    buffer.clear();
    int read;
    while ((read = in.read(buffer.array(), buffer.position(), buffer.remaining())) != -1) {
        buffer.position(buffer.position() + read);
    }
    buffer.flip();
}