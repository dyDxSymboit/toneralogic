if __name__ == "__main__":
    # This only runs if someone executes: python wsgi.py
    app.run(host="0.0.0.0", port=8000, debug=False)