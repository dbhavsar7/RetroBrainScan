import sqlite3
import json
from datetime import datetime
import base64


class DatabaseManager:
    def __init__(self, db_path="RetroBrainScanDB.db"):
        self.db_path = db_path

    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)

    def store_image(self, filename, image_data, metadata=None):
        """
        Store an image as JSON in the database
        
        Args:
            filename (str): Name of the image file
            image_data (bytes): Raw image data
            metadata (dict): Optional metadata about the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert image data to base64 for JSON storage
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare JSON payload
            json_data = {
                "filename": filename,
                "upload_timestamp": datetime.now().isoformat(),
                "image_data": image_base64,
                "metadata": metadata or {}
            }
            
            json_string = json.dumps(json_data)
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ImageData (filename, image_data, metadata)
                VALUES (?, ?, ?)
            ''', (filename, json_string, json.dumps(metadata or {})))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error storing image: {e}")
            return False

    def get_image(self, image_id):
        """
        Retrieve an image by ID
        
        Args:
            image_id (int): ID of the image in database
            
        Returns:
            dict: Image data and metadata, or None if not found
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, upload_timestamp, image_data, metadata
                FROM ImageData WHERE id = ?
            ''', (image_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "filename": row[1],
                    "upload_timestamp": row[2],
                    "image_data": row[3],  # JSON string
                    "metadata": row[4]     # JSON string
                }
            return None
            
        except Exception as e:
            print(f"Error retrieving image: {e}")
            return None

    def get_all_images(self, limit=100):
        """
        Retrieve all stored images
        
        Args:
            limit (int): Maximum number of images to retrieve
            
        Returns:
            list: List of image records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, filename, upload_timestamp
                FROM ImageData
                ORDER BY upload_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "filename": row[1],
                    "upload_timestamp": row[2]
                }
                for row in rows
            ]
            
        except Exception as e:
            print(f"Error retrieving images: {e}")
            return []

    def delete_image(self, image_id):
        """
        Delete an image by ID
        
        Args:
            image_id (int): ID of the image to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM ImageData WHERE id = ?', (image_id,))
            
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False

    def get_database_stats(self):
        """
        Get statistics about the database
        
        Returns:
            dict: Database statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM ImageData')
            total_images = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_images": total_images,
                "database_path": self.db_path
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_images": 0, "database_path": self.db_path}
