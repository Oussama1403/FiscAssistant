import { useCallback } from 'react';

import { useFetch } from '@/hooks/useFetch';

import {OllamaModel, OllamaModelDetail} from '@/types/ollama'


const useApiService = () => {
  const fetchService = useFetch();

  const getModels = useCallback(
    (): Promise<OllamaModel[]>  => {
      return fetchService.get(`/api/models`, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
    },
    [fetchService],
  );

  const getChatResponse = useCallback(
  (message: string, user_id = "default_user") => {
    return fetchService.post(`http://localhost:8000/chat`, {
      headers: {
        'Content-Type': 'application/json',
      },
      body: { message, user_id },
    });
  },
  [fetchService],
  );

  const getModelDetails = useCallback(
    (name: string) => {
      return fetchService.post(`/api/modeldetails`, {
        headers: {
          'Content-Type': 'application/json',
        },
        body: {name: name }, 
      });
    },
    [fetchService],
  );

  return {
    getModels,
    getModelDetails,
    getChatResponse
  };
};

export default useApiService;

